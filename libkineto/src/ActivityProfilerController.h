/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "ActivityLoggerFactory.h"
#include "CuptiActivityProfiler.h"
#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ConfigLoader.h"
#include "CuptiActivityApi.h"

namespace KINETO_NAMESPACE {

class Config;

class ActivityProfilerController : public ConfigLoader::ConfigHandler {
 public:
  explicit ActivityProfilerController(ConfigLoader& configLoader, bool cpuOnly);
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController();

  static void addLoggerFactory(
      const std::string& protocol,
      ActivityLoggerFactory::FactoryFunc factory);

  bool canAcceptConfig() override;
  void acceptConfig(const Config& config) override;

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

  void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) {
    profiler_->addChildActivityProfiler(std::move(profiler));
  }

  void addMetadata(const std::string& key, const std::string& value);

 private:
  void profilerLoop();

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;
  std::unique_ptr<CuptiActivityProfiler> profiler_;
  std::unique_ptr<ActivityLogger> logger_;
  std::thread* profilerThread_{nullptr};
  std::atomic_bool stopRunloop_{false};
  ConfigLoader& configLoader_;
};

} // namespace KINETO_NAMESPACE
