#pragma once

#include <libkineto.h>
#include <chrono>

#include "IActivityProfiler.h"
#include "Logger.h"
#include "output_base.h"

#include "KinetoDynamicPluginInterface.h"
#include "PluginTraceBuilder.h"
#include "PluginUtils.h"

namespace libkineto {

// This file handles pure C plugin profiler interface and converts to internal
// profiler interface

class PluginProfilerSession : public IActivityProfilerSession {
 public:
  PluginProfilerSession(
      const KinetoPlugin_ProfilerInterface& profiler,
      const std::string& name,
      const std::set<ActivityType>& enabled_activity_types)
      : name_(name),
        profiler_(profiler),
        enabled_activity_types_(enabled_activity_types) {
    KinetoPlugin_ProfilerCreate_Params createParams{
        KINETO_PLUGIN_PROFILER_CREATE_PARAMS_UNPADDED_STRUCT_SIZE};

    std::vector<KinetoPlugin_ProfileEventType> enabledActivityTypes =
        convertActivityTypeSet(enabled_activity_types_);
    createParams.pEnabledActivityTypes = enabledActivityTypes.data();
    createParams.enabledActivityTypesMaxLen = enabledActivityTypes.size();

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

  void pushCorrelationId(uint64_t id) override {
    KinetoPlugin_ProfilerPushCorrelationId_Params pushCorrelationIdParams{
        KINETO_PLUGIN_PROFILER_PUSH_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE};
    pushCorrelationIdParams.pProfilerHandle = pProfilerHandle_;
    pushCorrelationIdParams.correlationId = id;

    int errorCode =
        profiler_.profilerPushCorrelationId(&pushCorrelationIdParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerPushCorrelationId() with error "
                 << errorCode;
    }
  }

  void popCorrelationId() override {
    KinetoPlugin_ProfilerPopCorrelationId_Params popCorrelationIdParams{
        KINETO_PLUGIN_PROFILER_POP_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE};
    popCorrelationIdParams.pProfilerHandle = pProfilerHandle_;

    int errorCode = profiler_.profilerPopCorrelationId(&popCorrelationIdParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerPopCorrelationId() with error "
                 << errorCode;
    }
  }

  void pushUserCorrelationId(uint64_t id) override {
    KinetoPlugin_ProfilerPushUserCorrelationId_Params pushUserCorrelationIdParams{
        KINETO_PLUGIN_PROFILER_PUSH_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE};
    pushUserCorrelationIdParams.pProfilerHandle = pProfilerHandle_;
    pushUserCorrelationIdParams.userCorrelationId = id;

    int errorCode =
        profiler_.profilerPushUserCorrelationId(&pushUserCorrelationIdParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerPushUserCorrelationId() with error "
                 << errorCode;
    }
  }

  void popUserCorrelationId() override {
    KinetoPlugin_ProfilerPopUserCorrelationId_Params popUserCorrelationIdParams{
        KINETO_PLUGIN_PROFILER_POP_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE};
    popUserCorrelationIdParams.pProfilerHandle = pProfilerHandle_;

    int errorCode =
        profiler_.profilerPopUserCorrelationId(&popUserCorrelationIdParams);
    if (errorCode != 0) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " failed at profilerPopUserCorrelationId() with error "
                 << errorCode;
    }
  }

  TraceStatus status() {
    return status_;
  }

  // returns errors with this trace
  std::vector<std::string> errors() override {
    return {};
  }

  // processes trace activities using logger
  void processTrace(ActivityLogger& logger) override {
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
    deviceInfos_ = pluginTraceBuilder.getDeviceInfos();
    resourceInfos_ = pluginTraceBuilder.getResourceInfos();

    // Log events
    for (const auto& event : traceBuffer_->activities) {
      static_assert(
          std::is_same<
              std::remove_reference<decltype(event)>::type,
              const std::unique_ptr<GenericTraceActivity>>::value,
          "handleActivity is unsafe and relies on the caller to maintain not "
          "only lifetime but also address stability.");
      logger.handleActivity(*event);
    }

    return;
  }

  // returns device info used in this trace, could be nullptr
  std::unique_ptr<DeviceInfo> getDeviceInfo() override {
    if (deviceInfos_.empty()) {
      return {};
    }
    return std::make_unique<DeviceInfo>(deviceInfos_[0]);
  }

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
  std::vector<DeviceInfo> deviceInfos_;
  std::vector<ResourceInfo> resourceInfos_;
  const KinetoPlugin_ProfilerInterface profiler_;
  KinetoPlugin_ProfilerHandle* pProfilerHandle_ = nullptr;
  std::string name_;
  std::set<ActivityType> enabled_activity_types_;
  int64_t lastStartTimestampUtcNs_ = 0;
  int64_t lastStopTimestampUtcNs_ = 0;
};

class PluginProfiler : public IActivityProfiler {
 public:
  PluginProfiler(const KinetoPlugin_ProfilerInterface& profiler)
      : profiler_(profiler), isValid_(true) {
    isValid_ = validateProfiler();
    if (!isValid_) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " is not valid; skipping registration";
      return;
    }

    char profilerName[32];

    std::array<
        KinetoPlugin_ProfileEventType,
        KINETO_PLUGIN_PROFILE_EVENT_NUM_TYPES>
        supportedActivityTypes;
    std::fill(
        supportedActivityTypes.begin(),
        supportedActivityTypes.end(),
        KINETO_PLUGIN_PROFILE_EVENT_TYPE_INVALID);

    KinetoPlugin_ProfilerQuery_Params queryParams{
        KINETO_PLUGIN_PROFILER_QUERY_PARAMS_UNPADDED_STRUCT_SIZE};
    queryParams.pProfilerHandle = nullptr;
    queryParams.pProfilerName = &profilerName[0];
    queryParams.profilerNameMaxLen = 31;
    queryParams.pSupportedActivityTypes = &supportedActivityTypes[0];
    queryParams.supportedActivityTypesMaxLen = supportedActivityTypes.size();

    int errorCode = profiler_.profilerQuery(&queryParams);
    if (errorCode != 0) {
      name_.assign("N/A");
    } else {
      name_.assign(queryParams.pProfilerName);
    }

    supportedActivities_.clear();
    for (size_t i = 0; i < supportedActivityTypes.size(); i++) {
      if (supportedActivityTypes[i] !=
          KINETO_PLUGIN_PROFILE_EVENT_TYPE_INVALID) {
        supportedActivities_.insert(
            convertToActivityType(supportedActivityTypes[i]));
      }
    }
  }

  ~PluginProfiler() = default;

  const std::string& name() const override {
    return name_;
  }

  const std::set<ActivityType>& availableActivities() const override {
    return supportedActivities_;
  }

  std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const Config& /*config*/) override {
    // Check if profiler is valid
    if (!isValid_) {
      LOG(ERROR) << "Plugin profiler " << name_
                 << " is not valid, cannot configure";
      return nullptr;
    }

    // Check if the plugin supports ANY of the requested activity types
    // and compute the intersection of requested and supported types
    std::set<ActivityType> enabledTypes;
    for (const auto& activity_type : activity_types) {
      if (supportedActivities_.find(activity_type) !=
          supportedActivities_.end()) {
        enabledTypes.insert(activity_type);
      }
    }
    if (enabledTypes.empty()) {
      LOG(INFO) << "Plugin profiler " << name_
                << " does not support any of the requested activity types";
      return nullptr;
    }

    // [TODO] In future evolution of API we may want to pass in Config string or
    // a subset of config strings perhaps by searching for "PLUGIN_NAME_" in the
    // config string
    return std::make_unique<PluginProfilerSession>(
        profiler_, name_, enabledTypes);
  }

  std::unique_ptr<IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const Config& config) override {
    return configure(activity_types, config);
  }

 private:
  bool validateProfiler() {
    bool isValid = true;

    // Handle versioning
    // Currently expect the exact same version
    if (profiler_.unpaddedStructSize <
        KINETO_PLUGIN_PROFILER_INTERFACE_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Plugin profiler has an incompatible version";

      profiler_.profilerCreate =
          [](struct KinetoPlugin_ProfilerCreate_Params*) { return -1; };
      profiler_.profilerDestroy =
          [](struct KinetoPlugin_ProfilerDestroy_Params*) { return -1; };
      profiler_.profilerQuery = [](struct KinetoPlugin_ProfilerQuery_Params*) {
        return -1;
      };
      profiler_.profilerStart = [](struct KinetoPlugin_ProfilerStart_Params*) {
        return -1;
      };
      profiler_.profilerStop = [](struct KinetoPlugin_ProfilerStop_Params*) {
        return -1;
      };
      profiler_.profilerPushCorrelationId =
          [](struct KinetoPlugin_ProfilerPushCorrelationId_Params*) {
            return -1;
          };
      profiler_.profilerPopCorrelationId =
          [](struct KinetoPlugin_ProfilerPopCorrelationId_Params*) {
            return -1;
          };
      profiler_.profilerPushUserCorrelationId =
          [](struct KinetoPlugin_ProfilerPushUserCorrelationId_Params*) {
            return -1;
          };
      profiler_.profilerPopUserCorrelationId =
          [](struct KinetoPlugin_ProfilerPopUserCorrelationId_Params*) {
            return -1;
          };
      profiler_.profilerProcessEvents =
          [](struct KinetoPlugin_ProfilerProcessEvents_Params*) { return -1; };
      return false;
    }

    // Check if individual function pointers are implemented
    // For critical functions, set them to error-returning stubs if nullptr and
    // mark as invalid
    if (profiler_.profilerCreate == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerCreate is not implemented";
      profiler_.profilerCreate =
          [](struct KinetoPlugin_ProfilerCreate_Params*) { return -1; };
      isValid = false;
    }

    if (profiler_.profilerDestroy == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerDestroy is not implemented";
      profiler_.profilerDestroy =
          [](struct KinetoPlugin_ProfilerDestroy_Params*) { return -1; };
      isValid = false;
    }

    if (profiler_.profilerQuery == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerQuery is not implemented";
      profiler_.profilerQuery = [](struct KinetoPlugin_ProfilerQuery_Params*) {
        return -1;
      };
      isValid = false;
    }

    if (profiler_.profilerStart == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerStart is not implemented";
      profiler_.profilerStart = [](struct KinetoPlugin_ProfilerStart_Params*) {
        return -1;
      };
      isValid = false;
    }

    if (profiler_.profilerStop == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerStop is not implemented";
      profiler_.profilerStop = [](struct KinetoPlugin_ProfilerStop_Params*) {
        return -1;
      };
      isValid = false;
    }

    if (profiler_.profilerProcessEvents == nullptr) {
      LOG(ERROR) << "Plugin profiler profilerProcessEvents is not implemented";
      profiler_.profilerProcessEvents =
          [](struct KinetoPlugin_ProfilerProcessEvents_Params*) { return -1; };
      isValid = false;
    }

    // For correlation functions, provide default warning implementations if not
    // provided These don't affect validity
    if (profiler_.profilerPushCorrelationId == nullptr) {
      LOG(INFO) << "Plugin profiler profilerPushCorrelationId is not "
                   "implemented, using default stub";
      profiler_.profilerPushCorrelationId =
          [](struct KinetoPlugin_ProfilerPushCorrelationId_Params*) {
            LOG_FIRST_N(1, WARNING)
                << "profilerPushCorrelationId called but not "
                   "implemented by plugin";
            return 0;
          };
    }

    if (profiler_.profilerPopCorrelationId == nullptr) {
      LOG(INFO) << "Plugin profiler profilerPopCorrelationId is not "
                   "implemented, using default stub";
      profiler_.profilerPopCorrelationId =
          [](struct KinetoPlugin_ProfilerPopCorrelationId_Params*) {
            LOG_FIRST_N(1, WARNING)
                << "profilerPopCorrelationId called but not "
                   "implemented by plugin";
            return 0;
          };
    }

    if (profiler_.profilerPushUserCorrelationId == nullptr) {
      LOG(INFO) << "Plugin profiler profilerPushUserCorrelationId is not "
                   "implemented, using default stub";
      profiler_.profilerPushUserCorrelationId =
          [](struct KinetoPlugin_ProfilerPushUserCorrelationId_Params*) {
            LOG_FIRST_N(1, WARNING)
                << "profilerPushUserCorrelationId called but not "
                   "implemented by plugin";
            return 0;
          };
    }

    if (profiler_.profilerPopUserCorrelationId == nullptr) {
      LOG(INFO) << "Plugin profiler profilerPopUserCorrelationId is not "
                   "implemented, using default stub";
      profiler_.profilerPopUserCorrelationId =
          [](struct KinetoPlugin_ProfilerPopUserCorrelationId_Params*) {
            LOG_FIRST_N(1, WARNING)
                << "profilerPopUserCorrelationId called but not "
                   "implemented by plugin";
            return 0;
          };
    }

    return isValid;
  }

  KinetoPlugin_ProfilerInterface profiler_;
  std::string name_;
  std::set<ActivityType> supportedActivities_;
  bool isValid_;
};

} // namespace libkineto
