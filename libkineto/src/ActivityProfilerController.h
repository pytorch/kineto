// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityLoggerFactory.h"
#include "CuptiActivityProfiler.h"
#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ConfigLoader.h"
#include "CuptiActivityApi.h"
#include "LoggerCollector.h"

namespace KINETO_NAMESPACE {

class Config;

class ActivityProfilerController : public ConfigLoader::ConfigHandler {
 public:
  explicit ActivityProfilerController(ConfigLoader& configLoader, bool cpuOnly);
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController();

#if !USE_GOOGLE_LOG
  static void setLoggerCollectorFactory(
      std::function<std::unique_ptr<LoggerCollector>()> factory);
#endif // !USE_GOOGLE_LOG

  static void addLoggerFactory(
      const std::string& protocol,
      ActivityLoggerFactory::FactoryFunc factory);

  // These API are used for On-Demand Tracing.
  bool canAcceptConfig() override;
  void acceptConfig(const Config& config) override;
  void scheduleTrace(const Config& config);

  // These API are used for Synchronous Tracing.
  void prepareTrace(const Config& config);
  void startTrace();
  void step();
  std::unique_ptr<ActivityTraceInterface> stopTrace();

  bool isActive() {
    return profiler_->isActive();
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
  bool shouldActivateIterationConfig(int64_t currentIter);
  bool shouldActivateTimestampConfig(
      const std::chrono::time_point<std::chrono::system_clock>& now);
  void profilerLoop();
  void activateConfig(std::chrono::time_point<std::chrono::system_clock> now);

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;

  std::unique_ptr<CuptiActivityProfiler> profiler_;
  std::unique_ptr<ActivityLogger> logger_;
  std::thread* profilerThread_{nullptr};
  std::atomic_bool stopRunloop_{false};
  std::atomic<std::int64_t> iterationCount_{-1};
  ConfigLoader& configLoader_;
};

} // namespace KINETO_NAMESPACE
