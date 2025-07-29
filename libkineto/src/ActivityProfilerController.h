/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityLoggerFactory.h"
#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ConfigLoader.h"
#include "CuptiActivityApi.h"
#include "CuptiActivityProfiler.h"
#include "InvariantViolations.h"
#include "LoggerCollector.h"

namespace KINETO_NAMESPACE {
enum ThreadType {
  KINETO = 0,
  MEMORY_SNAPSHOT,
  THREAD_MAX_COUNT // Number of enum entries (used for array sizing)
};

class Config;

class ActivityProfilerController : public ConfigLoader::ConfigHandler {
 public:
  explicit ActivityProfilerController(ConfigLoader& configLoader, bool cpuOnly);
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController();

#if !USE_GOOGLE_LOG
  static std::shared_ptr<LoggerCollector> getLoggerCollector();
  static void setLoggerCollectorFactory(
      std::function<std::shared_ptr<LoggerCollector>()> factory);
#endif // !USE_GOOGLE_LOG

  static void addLoggerFactory(
      const std::string& protocol,
      ActivityLoggerFactory::FactoryFunc factory);

  static void setInvariantViolationsLoggerFactory(
      const std::function<std::unique_ptr<InvariantViolationsLogger>()>&
          factory);

  // These API are used for On-Demand Tracing.
  bool canAcceptConfig() override;
  void acceptConfig(const Config& config) override;
  void scheduleTrace(const Config& config);

  // These API are used for Synchronous Tracing.
  void prepareTrace(const Config& config);
  void toggleCollectionDynamic(const bool enable);
  void startTrace();
  void step();
  std::unique_ptr<ActivityTraceInterface> stopTrace();

  bool isActive() {
    return profiler_->isActive();
  }

  void transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
    return profiler_->transferCpuTrace(std::move(cpuTrace));
  }

  void recordThreadInfo() {
    profiler_->recordThreadInfo();
  }

  void addChildActivityProfiler(std::unique_ptr<IActivityProfiler> profiler) {
    profiler_->addChildActivityProfiler(std::move(profiler));
  }

  void addMetadata(const std::string& key, const std::string& value);

  void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id = "");

  void pushCorrelationId(uint64_t id) {
    profiler_->pushCorrelationId(id);
  }
  void popCorrelationId() {
    profiler_->popCorrelationId();
  }

  void pushUserCorrelationId(uint64_t id) {
    profiler_->pushUserCorrelationId(id);
  }
  void popUserCorrelationId() {
    profiler_->popUserCorrelationId();
  }

 private:
  bool shouldActivateIterationConfig(int64_t currentIter);
  bool shouldActivateTimestampConfig(
      const std::chrono::time_point<std::chrono::system_clock>& now);
  void profilerLoop();
  void memoryProfilerLoop();
  void activateConfig(std::chrono::time_point<std::chrono::system_clock> now);

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;

  std::unique_ptr<CuptiActivityProfiler> profiler_;
  std::unique_ptr<ActivityLogger> logger_;
  std::shared_ptr<LoggerCollector> loggerCollectorFactory_;
  std::thread* profilerThreads_[ThreadType::THREAD_MAX_COUNT] = {nullptr};
  std::atomic_bool stopRunloop_{false};
  std::atomic<std::int64_t> iterationCount_{-1};
  ConfigLoader& configLoader_;
};

} // namespace KINETO_NAMESPACE
