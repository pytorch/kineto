/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSessionV1.h"
#include "XpuptiActivityApiV2.h"

#include "time_since_epoch.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <iterator>

namespace KINETO_NAMESPACE {

using namespace std::literals::string_view_literals;

uint32_t XpuptiActivityProfilerSessionV1::iterationCount_ = 0;
std::vector<DeviceUUIDsT> XpuptiActivityProfilerSessionV1::deviceUUIDs_ = {};
std::unordered_set<std::string_view>
    XpuptiActivityProfilerSessionV1::correlateRuntimeOps_ = {
        "piextUSMEnqueueFill"sv,
        "urEnqueueUSMFill"sv,
        "piextUSMEnqueueFill2D"sv,
        "urEnqueueUSMFill2D"sv,
        "piextUSMEnqueueMemcpy"sv,
        "urEnqueueUSMMemcpy"sv,
        "piextUSMEnqueueMemset"sv,
        "piextUSMEnqueueMemcpy2D"sv,
        "urEnqueueUSMMemcpy2D"sv,
        "piextUSMEnqueueMemset2D"sv,
        "piEnqueueKernelLaunch"sv,
        "urEnqueueKernelLaunch"sv,
        "piextEnqueueKernelLaunchCustom"sv,
        "urEnqueueKernelLaunchCustomExp"sv,
        "piextEnqueueCooperativeKernelLaunch"sv,
        "urEnqueueCooperativeKernelLaunchExp"sv};

// =========== Session Constructor ============= //
XpuptiActivityProfilerSessionV1::XpuptiActivityProfilerSessionV1(
    XpuptiActivityApi& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : xpti_(xpti),
      name_(name),
      config_(config.clone()),
      activity_types_(activity_types) {
  enumDeviceUUIDs();
  xpti_.enableXpuptiActivities(activity_types_);
}

XpuptiActivityProfilerSessionV1::~XpuptiActivityProfilerSessionV1() {
  xpti_.clearActivities();
}

// =========== Session Public Methods ============= //
void XpuptiActivityProfilerSessionV1::start() {
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
}

void XpuptiActivityProfilerSessionV1::stop() {
  xpti_.disablePtiActivities(activity_types_);
  profilerEndTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
}

void XpuptiActivityProfilerSessionV1::toggleCollectionDynamic(
    const bool enable) {
  if (enable) {
    xpti_.enableXpuptiActivities(activity_types_);
  } else {
    xpti_.disablePtiActivities(activity_types_);
  }
}

void XpuptiActivityProfilerSessionV1::processTrace(ActivityLogger& logger) {
  traceBuffer_.span =
      libkineto::TraceSpan(profilerStartTs_, profilerEndTs_, name_);
  traceBuffer_.span.iteration = iterationCount_++;
  auto gpuBuffer = xpti_.activityBuffers();
  if (gpuBuffer) {
    xpti_.processActivities(
        *gpuBuffer,
        [this, &logger](const pti_view_record_base* record) -> void {
          handlePtiActivity(record, logger);
        });
  }
}

void XpuptiActivityProfilerSessionV1::processTrace(
    ActivityLogger& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

std::unique_ptr<libkineto::CpuTraceBuffer>
XpuptiActivityProfilerSessionV1::getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

void XpuptiActivityProfilerSessionV1::pushCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XpuptiActivityApi::CorrelationFlowType::Default);
}

void XpuptiActivityProfilerSessionV1::popCorrelationId() {
  xpti_.popCorrelationID(XpuptiActivityApi::CorrelationFlowType::Default);
}

void XpuptiActivityProfilerSessionV1::pushUserCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XpuptiActivityApi::CorrelationFlowType::User);
}

void XpuptiActivityProfilerSessionV1::popUserCorrelationId() {
  xpti_.popCorrelationID(XpuptiActivityApi::CorrelationFlowType::User);
}

void XpuptiActivityProfilerSessionV1::enumDeviceUUIDs() {
  if (!deviceUUIDs_.empty()) {
    return;
  }
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
          deviceUUIDs_.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr
              << "Warnings: UUID is not supported for this XPU device. The device index of records will be 0."
              << std::endl;
          deviceUUIDs_.push_back(DeviceUUIDsT{});
        }
      }
    }
  }
}

DeviceIndex_t XpuptiActivityProfilerSessionV1::getDeviceIdxFromUUID(
    const uint8_t deviceUUID[16]) {
  auto it = std::find_if(
      deviceUUIDs_.begin(),
      deviceUUIDs_.end(),
      [deviceUUID](const DeviceUUIDsT& deviceUUIDinVec) {
        return std::equal(
            deviceUUIDinVec.begin(),
            deviceUUIDinVec.end(),
            deviceUUID,
            deviceUUID + 16);
      });
  if (it == deviceUUIDs_.end()) {
    std::cerr
        << "Warnings: Can't find the legal XPU device from the given UUID."
        << std::endl;
    return static_cast<DeviceIndex_t>(0);
  }
  return static_cast<DeviceIndex_t>(std::distance(deviceUUIDs_.begin(), it));
}

} // namespace KINETO_NAMESPACE
