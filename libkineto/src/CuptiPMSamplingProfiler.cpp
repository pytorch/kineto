/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>

#if defined(HAS_CUPTI_PM_SAMPLING) && defined(CUDA_VERSION) && \
    CUDA_VERSION >= 12080

#include "CuptiPMSamplingProfiler.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <set>
#include <string>
#include <utility>

#include "ActivityType.h"
#include "Logger.h"
#include "libkineto.h"

namespace KINETO_NAMESPACE {
namespace {

constexpr char kTraceName[] = "CUPTI PM Sampling";
const std::string kProfilerName{kTraceName};
const std::set<libkineto::ActivityType> kSupportedActivities{
    libkineto::ActivityType::HARDWARE_COUNTERS};

int64_t realtimeTimestamp() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

} // namespace

CuptiPMSamplingSession::CuptiPMSamplingSession(CuptiPMSamplingConfig config)
    : deviceId_(config.deviceId), controller_(std::move(config)) {}

CuptiPMSamplingSession::~CuptiPMSamplingSession() = default;

bool CuptiPMSamplingSession::prepare() {
  return calibrateTimestampOffset() && controller_.prepare();
}

void CuptiPMSamplingSession::start() {
  if (!controller_.start()) {
    errors_.emplace_back("Failed to start CUPTI PM sampling");
    status_ = libkineto::TraceStatus::ERROR;
    return;
  }
  status_ = libkineto::TraceStatus::RECORDING;
}

void CuptiPMSamplingSession::stop() {
  controller_.stop();
  traceBuffer_ = buildTraceBuffer();
  if (status_ != libkineto::TraceStatus::ERROR) {
    status_ = libkineto::TraceStatus::PROCESSING;
  }
}

std::vector<std::string> CuptiPMSamplingSession::errors() {
  return errors_;
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

bool CuptiPMSamplingSession::calibrateTimestampOffset() {
  try {
    // PM samples and cuptiGetTimestamp share a nanosecond clock, so one
    // simultaneous reading maps both sample timestamps onto the trace clock.
    const auto before = realtimeTimestamp();
    const auto cuptiTimestamp = controller_.timestamp();
    const auto after = realtimeTimestamp();
    if (before < 0 || after < before || cuptiTimestamp == 0 ||
        cuptiTimestamp >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      LOG(WARNING) << "Failed to calibrate CUPTI PM sampling timestamps";
      return false;
    }

    const auto traceTimestamp = before + (after - before) / 2;
    timestampOffset_ = traceTimestamp - static_cast<int64_t>(cuptiTimestamp);
    return true;
  } catch (const std::exception& error) {
    LOG(WARNING) << "Failed to calibrate CUPTI PM sampling timestamps: "
                 << error.what();
    return false;
  }
}

std::optional<int64_t> CuptiPMSamplingSession::toTraceTimestamp(
    uint64_t timestamp) const {
  if (timestamp > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }

  const auto value = static_cast<int64_t>(timestamp);
  if ((timestampOffset_ > 0 &&
       value > std::numeric_limits<int64_t>::max() - timestampOffset_) ||
      (timestampOffset_ < 0 && value < -timestampOffset_)) {
    return std::nullopt;
  }
  return value + timestampOffset_;
}

std::unique_ptr<libkineto::CpuTraceBuffer> CuptiPMSamplingSession::
    buildTraceBuffer() const {
  auto buffer = std::make_unique<libkineto::CpuTraceBuffer>();
  buffer->span = libkineto::TraceSpan{0, 0, kTraceName};

  const auto samples = controller_.samples();
  const auto& metricNames = controller_.metricNames();
  bool hasActivities = false;
  for (const auto& sample : samples) {
    const auto start = toTraceTimestamp(sample.rawStartTimestamp);
    const auto end = toTraceTimestamp(sample.rawEndTimestamp);
    if (!start || !end) {
      LOG(WARNING)
          << "CUPTI PM sample timestamp is out of range after calibration";
      continue;
    }

    if (!hasActivities) {
      buffer->span.startTime = *start;
      buffer->span.endTime = *end;
      hasActivities = true;
    } else {
      buffer->span.startTime = std::min(buffer->span.startTime, *start);
      buffer->span.endTime = std::max(buffer->span.endTime, *end);
    }

    buffer->emplace_activity(
        buffer->span, libkineto::ActivityType::HARDWARE_COUNTERS, kTraceName);
    auto& activity = *buffer->activities.back();
    activity.startTime = *start;
    activity.endTime = *end;
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
  return configure(activityTypes, config);
}

} // namespace KINETO_NAMESPACE

#endif // HAS_CUPTI_PM_SAMPLING && CUDA_VERSION >= 12080
