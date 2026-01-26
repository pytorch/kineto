/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AbstractConfig.h"
#include "ActivityType.h"

#include <cassert>
#include <chrono>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace libkineto {

class Config : public AbstractConfig {
 public:
  Config();
  Config& operator=(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
  ~Config() override = default;

  // Return a full copy including feature config object
  [[nodiscard]] std::unique_ptr<Config> clone() const {
    auto cfg = std::unique_ptr<Config>(new Config(*this));
    cloneFeaturesInto(*cfg);
    return cfg;
  }

  bool handleOption(const std::string& name, std::string& val) override;

  void setClientDefaults() override;

  // Log events to this file
  [[nodiscard]] const std::string& eventLogFile() const {
    return eventLogFile_;
  }

  [[nodiscard]] bool activityProfilerEnabled() const {
    return activityProfilerEnabled_ ||
        activitiesOnDemandTimestamp_.time_since_epoch().count() > 0;
  }

  // Log activitiy trace to this file
  [[nodiscard]] const std::string& activitiesLogFile() const {
    return activitiesLogFile_;
  }

  // Log activitiy trace to this url
  [[nodiscard]] const std::string& activitiesLogUrl() const {
    return activitiesLogUrl_;
  }

  void setActivitiesLogUrl(const std::string& url) {
    activitiesLogUrl_ = url;
  }

  [[nodiscard]] bool activitiesLogToMemory() const {
    return activitiesLogToMemory_;
  }

  [[nodiscard]] bool eventProfilerEnabled() const {
    return !eventNames_.empty() || !metricNames_.empty();
  }

  // Is profiling enabled for the given device?
  [[nodiscard]] bool eventProfilerEnabledForDevice(uint32_t dev) const {
    return 0 != (eventProfilerDeviceMask_ & (1 << dev));
  }

  // Take a sample (read hardware counters) at this frequency.
  // This controls how often counters are read - if all counters cannot
  // be collected simultaneously then multiple samples are needed to
  // collect all requested counters - see multiplex period.
  [[nodiscard]] std::chrono::milliseconds samplePeriod() const {
    return samplePeriod_;
  }

  void setSamplePeriod(std::chrono::milliseconds period) {
    samplePeriod_ = period;
  }

  // When all requested counters cannot be collected simultaneously,
  // counters will be multiplexed at this frequency.
  // Multiplexing can have a large performance impact if done frequently.
  // To avoid a perf impact, keep this at 1s or above.
  [[nodiscard]] std::chrono::milliseconds multiplexPeriod() const {
    return multiplexPeriod_;
  }

  void setMultiplexPeriod(std::chrono::milliseconds period) {
    multiplexPeriod_ = period;
  }

  // Report counters at this frequency. Note that several samples can
  // be reported each time, see samplesPerReport.
  [[nodiscard]] std::chrono::milliseconds reportPeriod() const {
    return reportPeriod_;
  }

  void setReportPeriod(std::chrono::milliseconds msecs);

  // Number of samples dispatched each report period.
  // Must be in the range [1, report period / sample period].
  // In other words, aggregation is supported but not interpolation.
  [[nodiscard]] int samplesPerReport() const {
    return samplesPerReport_;
  }

  void setSamplesPerReport(int count) {
    samplesPerReport_ = count;
  }

  // The names of events to collect
  [[nodiscard]] const std::set<std::string>& eventNames() const {
    return eventNames_;
  }

  // Add additional events to be profiled
  void addEvents(const std::set<std::string>& names) {
    eventNames_.insert(names.begin(), names.end());
  }

  // The names of metrics to collect
  [[nodiscard]] const std::set<std::string>& metricNames() const {
    return metricNames_;
  }

  // Add additional metrics to be profiled
  void addMetrics(const std::set<std::string>& names) {
    metricNames_.insert(names.begin(), names.end());
  }

  [[nodiscard]] const std::vector<int>& percentiles() const {
    return eventReportPercentiles_;
  }

  // Profile for this long, then revert to base config
  [[nodiscard]] std::chrono::seconds eventProfilerOnDemandDuration() const {
    return eventProfilerOnDemandDuration_;
  }

  void setEventProfilerOnDemandDuration(std::chrono::seconds duration) {
    eventProfilerOnDemandDuration_ = duration;
  }

  // Too many event profilers on a single system can overload the driver.
  // At some point, latencies shoot through the roof and collection of samples
  // becomes impossible. To avoid this situation we have a limit of profilers
  // per GPU.
  // NOTE: Communication with a daemon is needed for this feature.
  // Library must be built with an active DaemonConfigLoader.
  [[nodiscard]] int maxEventProfilersPerGpu() const {
    return eventProfilerMaxInstancesPerGpu_;
  }

  // On Cuda11 we've seen occasional hangs when reprogramming counters
  // Monitor profiling threads and report when a thread is not responding
  // for a given number of seconds.
  // A period of 0 means disable.
  [[nodiscard]] std::chrono::seconds eventProfilerHeartbeatMonitorPeriod()
      const {
    return eventProfilerHeartbeatMonitorPeriod_;
  }

  // The types of activities selected in the configuration file
  [[nodiscard]] const std::set<ActivityType>& selectedActivityTypes() const {
    return selectedActivityTypes_;
  }

  // Set the types of activities to be traced
  [[nodiscard]] bool perThreadBufferEnabled() const {
    return perThreadBufferEnabled_;
  }

  void setSelectedActivityTypes(const std::set<ActivityType>& types) {
    selectedActivityTypes_ = types;
  }

  [[nodiscard]] bool isReportInputShapesEnabled() const {
    return enableReportInputShapes_;
  }

  [[nodiscard]] bool isProfileMemoryEnabled() const {
    return enableProfileMemory_;
  }

  [[nodiscard]] bool isWithStackEnabled() const {
    return enableWithStack_;
  }

  [[nodiscard]] bool isWithFlopsEnabled() const {
    return enableWithFlops_;
  }

  [[nodiscard]] bool isWithModulesEnabled() const {
    return enableWithModules_;
  }

  [[nodiscard]] bool isAdjustProfilerStepEnabled() const {
    return enableAdjustProfilerStep_;
  }

  // Trace for this long
  [[nodiscard]] std::chrono::milliseconds activitiesDuration() const {
    return activitiesDuration_;
  }

  // Trace for this many iterations, determined by external API
  [[nodiscard]] int activitiesRunIterations() const {
    return activitiesRunIterations_;
  }

  [[nodiscard]] int activitiesMaxGpuBufferSize() const {
    return activitiesMaxGpuBufferSize_;
  }

  [[nodiscard]] std::chrono::seconds activitiesWarmupDuration() const {
    return activitiesWarmupDuration_;
  }

  [[nodiscard]] int activitiesWarmupIterations() const {
    return activitiesWarmupIterations_;
  }

  // Show CUDA Synchronization Stream Wait Events
  [[nodiscard]] bool activitiesCudaSyncWaitEvents() const {
    return activitiesCudaSyncWaitEvents_;
  }

  void setActivitiesCudaSyncWaitEvents(bool enable) {
    activitiesCudaSyncWaitEvents_ = enable;
  }

  // Timestamp at which the profiling to start, requested by the user.
  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  requestTimestamp() const {
    if (profileStartTime_.time_since_epoch().count()) {
      return profileStartTime_;
    }
    // If no one requested timestamp, return 0.
    if (requestTimestamp_.time_since_epoch().count() == 0) {
      return requestTimestamp_;
    }

    // TODO(T94634890): Deprecate requestTimestamp
    return requestTimestamp_ + maxRequestAge() + activitiesWarmupDuration();
  }

  [[nodiscard]] bool hasProfileStartTime() const {
    return requestTimestamp_.time_since_epoch().count() > 0 ||
        profileStartTime_.time_since_epoch().count() > 0;
  }

  [[nodiscard]] int profileStartIteration() const {
    return profileStartIteration_;
  }

  [[nodiscard]] bool hasProfileStartIteration() const {
    return profileStartIteration_ >= 0 && activitiesRunIterations_ > 0;
  }

  void setProfileStartIteration(int iter) {
    profileStartIteration_ = iter;
  }

  [[nodiscard]] int profileStartIterationRoundUp() const {
    return profileStartIterationRoundUp_;
  }

  // calculate the start iteration accounting for warmup
  [[nodiscard]] int startIterationIncludingWarmup() const {
    if (!hasProfileStartIteration()) {
      return -1;
    }
    return profileStartIteration_ - activitiesWarmupIterations_;
  }

  [[nodiscard]] std::chrono::seconds maxRequestAge() const;

  // All VLOG* macros will log if the verbose log level is >=
  // the verbosity specified for the verbose log message.
  // Default value is -1, so messages with log level 0 will log by default.
  [[nodiscard]] int verboseLogLevel() const {
    return verboseLogLevel_;
  }

  // Modules for which verbose logging is enabled.
  // If empty, logging is enabled for all modules.
  [[nodiscard]] const std::vector<std::string>& verboseLogModules() const {
    return verboseLogModules_;
  }

  [[nodiscard]] bool sigUsr2Enabled() const {
    return enableSigUsr2_;
  }

  [[nodiscard]] bool ipcFabricEnabled() const {
    return enableIpcFabric_;
  }

  [[nodiscard]] std::chrono::seconds onDemandConfigUpdateIntervalSecs() const {
    return onDemandConfigUpdateIntervalSecs_;
  }

  static std::chrono::milliseconds alignUp(
      std::chrono::milliseconds duration,
      std::chrono::milliseconds alignment) {
    duration += alignment;
    return duration - (duration % alignment);
  }

  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  eventProfilerOnDemandStartTime() const {
    return eventProfilerOnDemandTimestamp_;
  }

  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  eventProfilerOnDemandEndTime() const {
    return eventProfilerOnDemandTimestamp_ + eventProfilerOnDemandDuration_;
  }

  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  activityProfilerRequestReceivedTime() const {
    return activitiesOnDemandTimestamp_;
  }

  static constexpr std::chrono::milliseconds kControllerIntervalMsecs{1000};

  // Users may request and set trace id and group trace id.
  [[nodiscard]] const std::string& requestTraceID() const {
    return requestTraceID_;
  }

  void setRequestTraceID(const std::string& tid) {
    requestTraceID_ = tid;
  }

  [[nodiscard]] const std::string& requestGroupTraceID() const {
    return requestGroupTraceID_;
  }

  void setRequestGroupTraceID(const std::string& gtid) {
    requestGroupTraceID_ = gtid;
  }

  [[nodiscard]] size_t cuptiDeviceBufferSize() const {
    return cuptiDeviceBufferSize_;
  }

  [[nodiscard]] size_t cuptiDeviceBufferPoolLimit() const {
    return cuptiDeviceBufferPoolLimit_;
  }

  [[nodiscard]] bool memoryProfilerEnabled() const {
    return memoryProfilerEnabled_;
  }

  [[nodiscard]] int profileMemoryDuration() const {
    return profileMemoryDuration_;
  }
  void updateActivityProfilerRequestReceivedTime();

  void printActivityProfilerConfig(std::ostream& s) const override;
  void setActivityDependentConfig() override;

  void validate(
      const std::chrono::time_point<std::chrono::system_clock>&
          fallbackProfileStartTime) override;

  static void addConfigFactory(
      std::string name,
      std::function<AbstractConfig*(Config&)> factory);

  void print(std::ostream& s) const;

  // Config relies on some state with global static lifetime. If other
  // threads are using the config, it's possible that the global state
  // is destroyed before the threads stop. By hanging onto this handle,
  // correct destruction order can be ensured.
  static std::shared_ptr<void> getStaticObjectsLifetimeHandle();

  [[nodiscard]] bool getTSCTimestampFlag() const {
    return useTSCTimestamp_;
  }

  void setTSCTimestampFlag(bool flag) {
    useTSCTimestamp_ = flag;
  }

  [[nodiscard]] const std::string& getCustomConfig() const {
    return customConfig_;
  }

  [[nodiscard]] uint32_t maxEvents() const {
    return maxEvents_;
  }

 private:
  explicit Config(const Config& other) = default;

  AbstractConfig* cloneDerived(AbstractConfig& parent) const override {
    // Clone from AbstractConfig not supported
    assert(false);
    return nullptr;
  }

  uint8_t createDeviceMask(const std::string& val);

  // Adds valid activity types from the user defined string list in the
  // configuration file
  void setActivityTypes(const std::vector<std::string>& selected_activities);

  // Sets the default activity types to be traced
  void selectDefaultActivityTypes() {
    // If the user has not specified an activity list, add all types
    for (ActivityType t : defaultActivityTypes()) {
      selectedActivityTypes_.insert(t);
    }
  }

  int verboseLogLevel_;
  std::vector<std::string> verboseLogModules_;

  // Event profiler
  // These settings are also supported in on-demand mode
  std::chrono::milliseconds samplePeriod_;
  std::chrono::milliseconds reportPeriod_;
  int samplesPerReport_;
  std::set<std::string> eventNames_;
  std::set<std::string> metricNames_;

  // On-demand duration
  std::chrono::seconds eventProfilerOnDemandDuration_;
  // Last on-demand request
  std::chrono::time_point<std::chrono::system_clock>
      eventProfilerOnDemandTimestamp_;

  int eventProfilerMaxInstancesPerGpu_;

  // Monitor whether event profiler threads are stuck
  // at this frequency
  std::chrono::seconds eventProfilerHeartbeatMonitorPeriod_;

  // These settings can not be changed on-demand
  std::string eventLogFile_;
  std::vector<int> eventReportPercentiles_ = {5, 25, 50, 75, 95};
  uint8_t eventProfilerDeviceMask_ = ~0;
  std::chrono::milliseconds multiplexPeriod_;

  // Activity profiler
  bool activityProfilerEnabled_;

  // Enable per-thread buffer
  bool perThreadBufferEnabled_;
  std::set<ActivityType> selectedActivityTypes_;

  // The activity profiler settings are all on-demand
  std::string activitiesLogFile_;

  std::string activitiesLogUrl_;

  // Log activities to memory buffer
  bool activitiesLogToMemory_{false};

  int activitiesMaxGpuBufferSize_;
  std::chrono::seconds activitiesWarmupDuration_;
  int activitiesWarmupIterations_;
  bool activitiesCudaSyncWaitEvents_;

  // Enable Profiler Config Options
  // Temporarily disable shape collection until we re-roll out the feature for
  // on-demand cases
  bool enableReportInputShapes_{false};
  bool enableProfileMemory_{false};
  bool enableWithStack_{false};
  bool enableWithFlops_{false};
  bool enableWithModules_{false};
  bool enableAdjustProfilerStep_{false};

  // Profile for specified iterations and duration
  std::chrono::milliseconds activitiesDuration_;
  int activitiesRunIterations_;

  // Below are not used
  // Use this net name for iteration count
  std::string activitiesExternalAPIIterationsTarget_;
  // Only profile nets that includes this in the name
  std::vector<std::string> activitiesExternalAPIFilter_;
  // Only profile nets with at least this many operators
  int activitiesExternalAPINetSizeThreshold_;
  // Only profile nets with at least this many GPU operators
  int activitiesExternalAPIGpuOpCountThreshold_;
  // Last activity profiler request
  std::chrono::time_point<std::chrono::system_clock>
      activitiesOnDemandTimestamp_;

  // ActivityProfilers are triggered by either:
  // Synchronized start timestamps
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  // Or start iterations.
  int profileStartIteration_;
  int profileStartIterationRoundUp_;

  // DEPRECATED
  std::chrono::time_point<std::chrono::system_clock> requestTimestamp_;

  // Enable profiling via SIGUSR2
  bool enableSigUsr2_;

  // Enable IPC Fabric instead of thrift communication
  bool enableIpcFabric_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;

  // Logger Metadata
  std::string requestTraceID_;
  std::string requestGroupTraceID_;

  // CUPTI Device Buffer
  size_t cuptiDeviceBufferSize_;
  size_t cuptiDeviceBufferPoolLimit_;

  // CUPTI Timestamp Format
  bool useTSCTimestamp_{true};

  // Memory Profiler
  bool memoryProfilerEnabled_{false};
  int profileMemoryDuration_{1000};

  // Used to flexibly configure some custom options, especially for custom
  // backends. How to parse this string is handled by the custom backend.
  std::string customConfig_;

  // Roctracer settings
  uint32_t maxEvents_{5000000};
};

constexpr char kUseDaemonEnvVar[] = "KINETO_USE_DAEMON";

bool isDaemonEnvVarSet();

// Returns a reference to the protobuf trace enabled flag.
// This allows the flag to be set externally (e.g., from JustKnobs in FBConfig)
// and read in other components (e.g., ChromeTraceLogger).
bool& get_protobuf_trace_enabled();

} // namespace libkineto
