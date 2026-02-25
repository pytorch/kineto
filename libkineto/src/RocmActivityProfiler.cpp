/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef HAS_ROCTRACER

#include "RocmActivityProfiler.h"
#include <fmt/format.h>
#include <roctracer.h>
#include <string>
#include "DeviceUtil.h"

#include "ActivityBuffers.h"
#include "Config.h"
#include "Logger.h"
// RoctracerActivity.h must stay in this .cpp only — RoctracerActivity_inl.h
// has inline functions referencing thread_local anonymous-namespace maps
#include "RoctracerActivity.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using std::string;

namespace KINETO_NAMESPACE {

RocmActivityProfiler::RocmActivityProfiler(
    RoctracerActivityApi& roctracer,
    bool cpuOnly)
    : GenericActivityProfiler(cpuOnly), roctracer_(roctracer) {
  if (isGpuAvailable()) {
    logGpuVersions();
  }
}

void RocmActivityProfiler::logGpuVersions() {
  uint32_t majorVersion = roctracer_version_major();
  uint32_t minorVersion = roctracer_version_minor();
  std::string roctracerVersion =
      std::to_string(majorVersion) + "." + std::to_string(minorVersion);
  int hipRuntimeVersion = 0, hipDriverVersion = 0;
  CUDA_CALL(hipRuntimeGetVersion(&hipRuntimeVersion));
  CUDA_CALL(hipDriverGetVersion(&hipDriverVersion));
  LOG(INFO) << "HIP versions. Roctracer: " << roctracerVersion
            << "; Runtime: " << hipRuntimeVersion
            << "; Driver: " << hipDriverVersion;

  LOGGER_OBSERVER_ADD_METADATA("roctracer_version", roctracerVersion);
  LOGGER_OBSERVER_ADD_METADATA(
      "hip_runtime_version", std::to_string(hipRuntimeVersion));
  LOGGER_OBSERVER_ADD_METADATA(
      "hip_driver_version", std::to_string(hipDriverVersion));
  addVersionMetadata("roctracer_version", roctracerVersion);
  addVersionMetadata("hip_runtime_version", std::to_string(hipRuntimeVersion));
  addVersionMetadata("hip_driver_version", std::to_string(hipDriverVersion));
}

void RocmActivityProfiler::setMaxGpuBufferSize(int size) {
  roctracer_.setMaxBufferSize(size);
}

void RocmActivityProfiler::enableGpuTracing() {
  roctracer_.setMaxEvents(config().maxEvents());
  roctracer_.enableActivities(derivedConfig_->profileActivityTypes());
}

void RocmActivityProfiler::disableGpuTracing() {
  roctracer_.disableActivities(derivedConfig_->profileActivityTypes());
}

void RocmActivityProfiler::clearGpuActivities() {
  roctracer_.clearActivities();
}

bool RocmActivityProfiler::isGpuCollectionStopped() const {
  return roctracer_.stopCollection;
}

void RocmActivityProfiler::synchronizeGpuDevice() {
  CUDA_CALL(hipDeviceSynchronize());
  roctracer_.flushActivities();
}

void RocmActivityProfiler::pushCorrelationIdImpl(
    uint64_t id,
    CorrelationFlowType type) {
  RoctracerActivityApi::CorrelationFlowType roctracerType =
      (type == CorrelationFlowType::User)
      ? RoctracerActivityApi::CorrelationFlowType::User
      : RoctracerActivityApi::CorrelationFlowType::Default;
  RoctracerActivityApi::pushCorrelationID(id, roctracerType);
}

void RocmActivityProfiler::popCorrelationIdImpl(CorrelationFlowType type) {
  RoctracerActivityApi::CorrelationFlowType roctracerType =
      (type == CorrelationFlowType::User)
      ? RoctracerActivityApi::CorrelationFlowType::User
      : RoctracerActivityApi::CorrelationFlowType::Default;
  RoctracerActivityApi::popCorrelationID(roctracerType);
}

void RocmActivityProfiler::onResetTraceData() {
  roctracer_.teardownContext();
}

void RocmActivityProfiler::onFinalizeTrace(
    const Config& /*config*/,
    ActivityLogger& /*logger*/) {
  // No additional overhead info for ROCm currently
}

void RocmActivityProfiler::processGpuActivities(ActivityLogger& logger) {
  VLOG(0) << "Retrieving GPU activity buffers";
  const int count = roctracer_.processActivities(
      std::bind(
          &RocmActivityProfiler::handleRoctracerActivity,
          this,
          std::placeholders::_1,
          &logger),
      std::bind(
          &RocmActivityProfiler::handleCorrelationActivity,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3));
  LOG(INFO) << "Processed " << count << " GPU records";
  LOGGER_OBSERVER_ADD_EVENT_COUNT(count);
}

inline void RocmActivityProfiler::handleCorrelationActivity(
    uint64_t correlationId,
    uint64_t externalId,
    RoctracerLogger::CorrelationDomain externalKind) {
  if (externalKind == RoctracerLogger::CorrelationDomain::Domain0) {
    cpuCorrelationMap_[correlationId] = externalId;
  } else if (externalKind == RoctracerLogger::CorrelationDomain::Domain1) {
    userCorrelationMap_[correlationId] = externalId;
  } else {
    LOG(WARNING)
        << "Invalid RoctracerLogger::CorrelationDomain sent to handleCorrelationActivity";
    ecs_.invalid_external_correlation_events++;
  }
}

template <class T>
void RocmActivityProfiler::handleRuntimeActivity(
    const T* activity,
    ActivityLogger* logger) {
  int32_t tid = activity->tid;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->id, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity<T>(activity, linked));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGpuActivityPresent(true);
}

inline void RocmActivityProfiler::handleGpuActivity(
    const roctracerAsyncRow* act,
    ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(act->id, cpuCorrelationMap_);
  const auto& gpu_activity =
      traceBuffers_->addActivityWrapper(GpuActivity(act, linked));
  GenericActivityProfiler::handleGpuActivity(gpu_activity, logger);
}

void RocmActivityProfiler::handleRoctracerActivity(
    const roctracerBase* record,
    ActivityLogger* logger) {
  switch (record->type) {
    case ROCTRACER_ACTIVITY_DEFAULT:
      handleRuntimeActivity(
          reinterpret_cast<const roctracerRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_KERNEL:
      handleRuntimeActivity(
          reinterpret_cast<const roctracerKernelRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_COPY:
      handleRuntimeActivity(
          reinterpret_cast<const roctracerCopyRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_MALLOC:
      handleRuntimeActivity(
          reinterpret_cast<const roctracerMallocRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_ASYNC:
      handleGpuActivity(
          reinterpret_cast<const roctracerAsyncRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_NONE:
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->type;
      ecs_.unexepected_cuda_events++;
      break;
  }
}

} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
