/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "ActivityProfiler.h"
#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "CuptiActivityInterface.h"

namespace KINETO_NAMESPACE {

class Config;

using ActivityLoggerFactory =
    std::function<std::unique_ptr<ActivityLogger>(const Config&)>;

class ActivityProfilerController {
 public:
  explicit ActivityProfilerController(bool cpuOnly);
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController();

  static void setLoggerFactory(const ActivityLoggerFactory& factory);

  void scheduleTrace(const Config& config);

  void prepareTrace(const Config& config);

  void startTrace() {
    profiler_->startTrace(std::chrono::system_clock::now());
  }

  std::unique_ptr<ActivityTraceInterface> stopTrace();

  bool isActive() {
    return profiler_->isActive();
  }

  bool traceInclusionFilter(const std::string& match) {
    return profiler_->applyNetFilter(match);
  }

  void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
    return profiler_->transferCpuTrace(std::move(cpuTrace));
  }

  void recordThreadInfo() {
    profiler_->recordThreadInfo();
  }

  void addMetadata(const std::string& key, const std::string& value);

 private:
  void profilerLoop();

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;
  std::unique_ptr<ActivityProfiler> profiler_;
  std::unique_ptr<ActivityLogger> logger_;
  std::thread* profilerThread_{nullptr};
  std::atomic_bool stopRunloop_{false};
};

} // namespace KINETO_NAMESPACE
