/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "CuptiPMSamplingController.h"
#include "IActivityProfiler.h"

namespace libkineto {
struct CpuTraceBuffer;
}

namespace KINETO_NAMESPACE {

class CuptiPMSamplingSession final
    : public libkineto::IActivityProfilerSession {
 public:
  explicit CuptiPMSamplingSession(CuptiPMSamplingConfig config);
  ~CuptiPMSamplingSession() override;

  [[nodiscard]] bool prepare();
  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  [[nodiscard]] std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  [[nodiscard]] std::vector<libkineto::ResourceInfo> getResourceInfos()
      override;
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer()
      override;

 private:
  [[nodiscard]] int64_t toTraceTimestamp(uint64_t timestamp) const;
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> buildTraceBuffer()
      const;

  int32_t deviceId_;
  CuptiPMSamplingController controller_;
  std::unique_ptr<libkineto::CpuTraceBuffer> traceBuffer_;
  std::vector<std::string> errors_;
};

// The naming is unfortunate here, because CUPTI PM sampling is not activity
// profiling. However, Kineto's registration logic is based on
// IActivityProfiler, so this is the simplest way to introduce a new profiler.
class CuptiPMSamplingProfiler final : public libkineto::IActivityProfiler {
 public:
  explicit CuptiPMSamplingProfiler(CuptiPMSamplingConfig config);

  [[nodiscard]] const std::string& name() const override;
  [[nodiscard]] const std::set<libkineto::ActivityType>& availableActivities()
      const override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t startTimeMs,
      int64_t durationMs,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

 private:
  CuptiPMSamplingConfig config_;
};

} // namespace KINETO_NAMESPACE
