/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiPMSamplingProfiler.h"

#include <algorithm>
#include <set>
#include <string>
#include <utility>

#include "ActivityType.h"
#include "CuptiTimestamp.h"
#include "Logger.h"
#include "libkineto.h"

namespace KINETO_NAMESPACE {
namespace {

const std::string kProfilerName{"CUPTI PM Sampling"};
const std::set<libkineto::ActivityType> kSupportedActivities{
    libkineto::ActivityType::HARDWARE_COUNTERS};

} // namespace

CuptiPMSamplingSession::CuptiPMSamplingSession(CuptiPMSamplingConfig config)
    : deviceId_(config.deviceId), controller_(std::move(config)) {}

CuptiPMSamplingSession::~CuptiPMSamplingSession() = default;

bool CuptiPMSamplingSession::prepare() {
  // PM sampling and activity profiling share the same timestamp callbacks.
  // Currently, the PM sampling profiler is registered as a child profiler and
  // prepare() always runs after activity profiling enables the callbacks in
  // enableGpuTracing
  if (!isCuptiTimestampSourceReady()) {
    LOG(WARNING) << "CUPTI PM sampling requires timestamp setup from the "
                    "CUPTI Activity profiler";
    return false;
  }
  return controller_.prepare();
}

void CuptiPMSamplingSession::start() {
  controller_.start();
}

void CuptiPMSamplingSession::stop() {
  controller_.stop();
  traceBuffer_ = buildTraceBuffer();
}

std::vector<std::string> CuptiPMSamplingSession::errors() {
  // GenericActivityProfiler does not consume child profiler status or errors.
  // The controller logs failures directly for visibility.
  return {};
}

void CuptiPMSamplingSession::processTrace(libkineto::ActivityLogger& logger) {
  if (!traceBuffer_) {
    return;
  }
  for (const auto& activity : traceBuffer_->activities) {
    libkineto::CpuTraceBuffer::toRef(activity).log(logger);
  }
}

std::unique_ptr<libkineto::DeviceInfo> CuptiPMSamplingSession::getDeviceInfo() {
  return nullptr;
}

std::vector<libkineto::ResourceInfo> CuptiPMSamplingSession::
    getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> CuptiPMSamplingSession::
    getTraceBuffer() {
  return std::move(traceBuffer_);
}

std::unique_ptr<libkineto::CpuTraceBuffer> CuptiPMSamplingSession::
    buildTraceBuffer() {
  auto buffer = std::make_unique<libkineto::CpuTraceBuffer>();
  buffer->span = libkineto::TraceSpan{0, 0, kProfilerName};

  const auto samples = controller_.takeSamples();
  const auto& metricNames = controller_.metricNames();
  bool hasActivities = false;
  for (const auto& sample : samples) {
    const auto start = convertCuptiTimestamp(sample.rawStartTimestamp);
    const auto end = convertCuptiTimestamp(sample.rawEndTimestamp);

    if (!hasActivities) {
      buffer->span.startTime = start;
      buffer->span.endTime = end;
      hasActivities = true;
    } else {
      buffer->span.startTime = std::min(buffer->span.startTime, start);
      buffer->span.endTime = std::max(buffer->span.endTime, end);
    }

    buffer->emplace_activity(
        buffer->span,
        libkineto::ActivityType::HARDWARE_COUNTERS,
        kProfilerName);
    auto& activity = *buffer->activities.back();
    activity.startTime = start;
    activity.endTime = end;
    activity.device = deviceId_;
    for (size_t i = 0; i < metricNames.size(); ++i) {
      activity.addCounterValue(metricNames[i], sample.values[i]);
    }
  }

  return buffer;
}

CuptiPMSamplingProfiler::CuptiPMSamplingProfiler(CuptiPMSamplingConfig config)
    : config_(std::move(config)) {}

const std::string& CuptiPMSamplingProfiler::name() const {
  return kProfilerName;
}

const std::set<libkineto::ActivityType>& CuptiPMSamplingProfiler::
    availableActivities() const {
  return kSupportedActivities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> CuptiPMSamplingProfiler::
    configure(
        const std::set<libkineto::ActivityType>& activityTypes,
        const libkineto::Config& /*config*/) {
  if (activityTypes.find(libkineto::ActivityType::HARDWARE_COUNTERS) ==
      activityTypes.end()) {
    return nullptr;
  }

  auto session = std::make_unique<CuptiPMSamplingSession>(config_);
  if (!session->prepare()) {
    return nullptr;
  }
  return session;
}

std::unique_ptr<libkineto::IActivityProfilerSession> CuptiPMSamplingProfiler::
    configure(
        int64_t /*startTimeMs*/,
        int64_t /*durationMs*/,
        const std::set<libkineto::ActivityType>& activityTypes,
        const libkineto::Config& config) {
  // GenericActivityProfiler calls this overload for both synchronous and
  // asynchronous traces. PM sampling does not schedule itself; the parent
  // drives collection through start() and stop(), so the timing arguments are
  // intentionally unused.
  return configure(activityTypes, config);
}

} // namespace KINETO_NAMESPACE
