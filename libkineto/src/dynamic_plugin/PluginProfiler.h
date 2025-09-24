#pragma once

#include <chrono>
#include <libkineto.h>

#include "IActivityProfiler.h"
#include "Logger.h"
#include "output_base.h"

#include "DynamicPluginInterface.h"
#include "PluginTraceBuilder.h"

namespace libkineto {

// This file handles pure C plugin profiler interface and converts to internal
// profiler interface

class PluginProfilerSession : public IActivityProfilerSession {

public:
  PluginProfilerSession(const KinetoPlugin_ProfilerInterface &profiler,
                        const std::string &name)
      : name_(name), profiler_(profiler) {
    KinetoPlugin_ProfilerCreate_Params createParams{
        KINETO_PLUGIN_PROFILER_CREATE_PARAMS_UNPADDED_STRUCT_SIZE};

    int errorCode = profiler_.profilerCreate(&createParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerCreate() with error " << errorCode;
      pProfilerHandle_ = nullptr;
    } else {
      pProfilerHandle_ = createParams.pProfilerHandle;
    }
  }

  ~PluginProfilerSession() {
    if (pProfilerHandle_ == nullptr) {
      return;
    }

    KinetoPlugin_ProfilerDestroy_Params destroyParams{
        KINETO_PLUGIN_PROFILER_DESTROY_PARAMS_UNPADDED_STRUCT_SIZE};
    destroyParams.pProfilerHandle = pProfilerHandle_;

    int errorCode = profiler_.profilerDestroy(&destroyParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerDestroy() with error " << errorCode;
    }
  }

  // start the trace collection synchronously
  void start() override {
    lastStartTimestampUtcNs_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    if (pProfilerHandle_ == nullptr) {
      return;
    }

    KinetoPlugin_ProfilerStart_Params startParams{
        KINETO_PLUGIN_PROFILER_START_PARAMS_UNPADDED_STRUCT_SIZE};
    startParams.pProfilerHandle = pProfilerHandle_;

    int errorCode = profiler_.profilerStart(&startParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerStart() with error " << errorCode;
    }
  }

  // stop the trace collection synchronously
  void stop() override {
    lastStopTimestampUtcNs_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    if (pProfilerHandle_ == nullptr) {
      return;
    }

    KinetoPlugin_ProfilerStop_Params stopParams{
        KINETO_PLUGIN_PROFILER_STOP_PARAMS_UNPADDED_STRUCT_SIZE};
    stopParams.pProfilerHandle = pProfilerHandle_;

    int errorCode = profiler_.profilerStop(&stopParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerStop() with error " << errorCode;
    }
  }

  TraceStatus status() { return status_; }

  // returns errors with this trace
  std::vector<std::string> errors() override { return {}; }

  // processes trace activities using logger
  void processTrace(ActivityLogger &logger) override {
    if (pProfilerHandle_ == nullptr) {
      return;
    }

    auto traceSpan =
        TraceSpan(lastStartTimestampUtcNs_, lastStopTimestampUtcNs_, name_);
    PluginTraceBuilder pluginTraceBuilder{traceSpan};
    const KinetoPlugin_TraceBuilder traceBuilder =
        pluginTraceBuilder.toCTraceBuilder();

    KinetoPlugin_ProfilerProcessEvents_Params profilerProcessEventsParams{
        KINETO_PLUGIN_PROFILER_PROCESS_EVENTS_PARAMS_UNPADDED_STRUCT_SIZE};
    profilerProcessEventsParams.pProfilerHandle = pProfilerHandle_;
    profilerProcessEventsParams.pTraceBuilder = &traceBuilder;

    int errorCode =
        profiler_.profilerProcessEvents(&profilerProcessEventsParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerProcessEvents() with error "
                 << errorCode;
    }

    // Take ownership of trace buffer from builder
    traceBuffer_ = pluginTraceBuilder.getTraceBuffer();
    resourceInfos_ = pluginTraceBuilder.getResourceInfos();

    // Log events
    for (const auto &event : traceBuffer_->activities) {
      static_assert(
          std::is_same<std::remove_reference<decltype(event)>::type,
                       const std::unique_ptr<GenericTraceActivity>>::value,
          "handleActivity is unsafe and relies on the caller to maintain not "
          "only lifetime but also address stability.");
      logger.handleActivity(*event);
    }

    return;
  }

  // returns device info used in this trace, could be nullptr
  std::unique_ptr<DeviceInfo> getDeviceInfo() override { return {}; }

  // returns resource info used in this trace, could be empty
  std::vector<ResourceInfo> getResourceInfos() override {
    return resourceInfos_;
  }

  // release ownership of the trace events and metadata
  std::unique_ptr<CpuTraceBuffer> getTraceBuffer() override {
    return std::move(traceBuffer_);
  }

private:
  std::unique_ptr<CpuTraceBuffer> traceBuffer_;
  std::vector<ResourceInfo> resourceInfos_;
  const KinetoPlugin_ProfilerInterface profiler_;
  KinetoPlugin_ProfilerHandle *pProfilerHandle_ = nullptr;
  std::string name_;
  int64_t lastStartTimestampUtcNs_ = 0;
  int64_t lastStopTimestampUtcNs_ = 0;
};

class PluginProfiler : public IActivityProfiler {

public:
  PluginProfiler(const KinetoPlugin_ProfilerInterface &profiler)
      : profiler_(profiler) {
    validateProfiler();

    char profilerName[32];
    KinetoPlugin_ProfilerQuery_Params queryParams{
        KINETO_PLUGIN_PROFILER_QUERY_PARAMS_UNPADDED_STRUCT_SIZE};
    queryParams.pProfilerHandle = nullptr;
    queryParams.pProfilerName = &profilerName[0];
    queryParams.profilerNameMaxLen = 31;

    int errorCode = profiler_.profilerQuery(&queryParams);
    if (errorCode != 0) {
      name_.assign("N/A");
    } else {
      name_.assign(queryParams.pProfilerName);
    }

    // [TODO] Query plugin for available activities
  }

  ~PluginProfiler() {}

  const std::string &name() const override { return name_; }

  const std::set<ActivityType> &availableActivities() const override {
    // [TODO] Fix below
    static const std::set<ActivityType> supported{
        ActivityType::CUDA_PROFILER_RANGE};
    return supported;
  }

  std::unique_ptr<IActivityProfilerSession>
  configure(const std::set<ActivityType> &activity_types,
            const Config &config) override {
    // [TODO] Check activity types to determine if creating session or nullptr
    return std::make_unique<PluginProfilerSession>(profiler_, name_);
  }

  std::unique_ptr<IActivityProfilerSession>
  configure(int64_t ts_ms, int64_t duration_ms,
            const std::set<ActivityType> &activity_types,
            const Config &config) override {
    return configure(activity_types, config);
  }

private:
  void validateProfiler() {
    // Handle versioning
    // Currently expect the exact same version
    if (profiler_.unpaddedStructSize <
        KINETO_PLUGIN_PROFILER_INTERFACE_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Plugin profiler has an incompatible version";

      profiler_.profilerCreate =
          [](struct KinetoPlugin_ProfilerCreate_Params *) { return -1; };
      profiler_.profilerDestroy =
          [](struct KinetoPlugin_ProfilerDestroy_Params *) { return -1; };
      profiler_.profilerQuery = [](struct KinetoPlugin_ProfilerQuery_Params *) {
        return -1;
      };
      profiler_.profilerStart = [](struct KinetoPlugin_ProfilerStart_Params *) {
        return -1;
      };
      profiler_.profilerStop = [](struct KinetoPlugin_ProfilerStop_Params *) {
        return -1;
      };
      profiler_.profilerProcessEvents =
          [](struct KinetoPlugin_ProfilerProcessEvents_Params *) { return -1; };
    }
  }

  KinetoPlugin_ProfilerInterface profiler_;
  std::string name_;
};

} // namespace libkineto
