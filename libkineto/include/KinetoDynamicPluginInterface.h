#ifndef KINETO_DYNAMIC_PLUGIN_FORMAT_H
#define KINETO_DYNAMIC_PLUGIN_FORMAT_H

#include <stdint.h>

#define KINETO_PLUGIN_LIB_DIR_PATH_ENV_VARIABLE "KINETO_PLUGIN_LIB_DIR_PATH"

// NOTE: This file uses unpadded struct size for version control
// To support backward compatibility, do NOT remove, reorder, or resize the
// existing fields For details, visit
// https://github.com/annarev/community/commit/09e231324c3b2b1f653c3f385fd2120edd460815
#define KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(struct_type, last_field_name) \
  (size_t)(offsetof(struct_type, last_field_name) +                      \
           sizeof(((struct_type*)1000)->last_field_name))

#ifdef __cplusplus
extern "C" {
#endif

// Event types that can be supported via plugin. Please sync
// with ActivityType.h if you add new event types.
enum KinetoPlugin_ProfileEventType {
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_INVALID = 0,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_OP, // cpu side ops
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_USER_ANNOTATION,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_USER_ANNOTATION,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMCPY,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMSET,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL, // on-device kernels
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_EXTERNAL_CORRELATION,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME, // host side cuda runtime
                                                 // events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_DRIVER, // host side cuda driver events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_INSTANT_EVENT, // host side point-like
                                                      // events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_PYTHON_FUNCTION,
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_OVERHEAD, // CUPTI induced overhead events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_RUNTIME, // host side MTIA runtime
                                                 // events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_CCP_EVENTS, // MTIA ondevice CCP events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_INSIGHT, // MTIA Insight Events
                                                 // sampled from its overhead
                                                 // API.
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_SYNC, // synchronization events between
                                              // runtime and kernels
  // Optional Activity types
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_GLOW_RUNTIME, // host side glow runtime
                                                 // events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_PROFILER_RANGE, // CUPTI Profiler range
                                                        // for performance
                                                        // metrics
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_HPU_OP, // HPU host side runtime event
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_XPU_RUNTIME, // host side xpu runtime events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_COLLECTIVE_COMM, // collective communication
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_PM_COUNTER, // GPU performance monitoring
                                                   // counter

  // PRIVATEUSE1 Activity types are used for custom backends.
  // The corresponding device type is `DeviceType::PrivateUse1` in PyTorch.
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_RUNTIME, // host side privateUse1
                                                        // runtime events
  KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_DRIVER, // host side privateUse1
                                                       // driver events

  KINETO_PLUGIN_PROFILE_EVENT_NUM_TYPES
};

struct KinetoPlugin_ProfileEvent {
  // Always set to KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;
  // Event type
  KinetoPlugin_ProfileEventType eventType;
  // Start timestamp in UTC nanoseconds
  int64_t startTimeUtcNs;
  // End timestamp in UTC nanoseconds
  int64_t endTimeUtcNs;
  // Event ID
  int64_t eventId;
  // If CPU, equivalent to process ID
  union {
    int32_t deviceId;
    int32_t processId;
  };
  // If CPU, equivalent to thread ID
  union {
    int32_t resourceId;
    int32_t threadId;
  };
};
#define KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(struct KinetoPlugin_ProfileEvent, threadId)

enum KinetoPlugin_ProfileEventFlowType {
  KINETO_PLUGIN_PROFILE_EVENT_FLOW_TYPE_INVALID = 0,
  KINETO_PLUGIN_PROFILE_EVENT_FLOW_TYPE_FWD_BWD,
  KINETO_PLUGIN_PROFILE_EVENT_FLOW_TYPE_ASYNC_CPU_GPU,
  KINETO_PLUGIN_PROFILE_EVENT_NUM_FLOW_TYPES
};

struct KinetoPlugin_ProfileEventFlow {
  // Always set to KINETO_PLUGIN_PROFILE_EVENT_FLOW_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;
  // Flow type
  KinetoPlugin_ProfileEventFlowType flowType;
  // Use the same non-zero ID to connect two events
  uint32_t flowId;
  // Set to true if flow is a starting point
  bool isStartPoint;
};
#define KINETO_PLUGIN_PROFILE_EVENT_FLOW_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                         \
      struct KinetoPlugin_ProfileEventFlow, isStartPoint)

struct KinetoPlugin_ProfileDeviceInfo {
  // Always set to KINETO_PLUGIN_PROFILE_DEVICE_INFO_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;
  // Device ID
  int64_t deviceId;
  // Device sort index
  int64_t sortIndex;
  // Device name
  const char* pName;
  // Device label
  const char* pLabel;
};
#define KINETO_PLUGIN_PROFILE_DEVICE_INFO_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                          \
      struct KinetoPlugin_ProfileDeviceInfo, pLabel)
struct KinetoPlugin_ProfileResourceInfo {
  // Always set to KINETO_PLUGIN_PROFILE_RESOURCE_INFO_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;
  // If CPU, equivalent to process ID
  union {
    int32_t deviceId;
    int32_t processId;
  };
  // If CPU, equivalent to thread ID
  union {
    int32_t resourceId;
    int32_t threadId;
  };
  // Lower value is displayed more towards top
  int32_t displayOrder;
  // Textual display in the viewer (ephemeral pointer)
  const char* pName;
};
#define KINETO_PLUGIN_PROFILE_RESOURCE_INFO_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                            \
      struct KinetoPlugin_ProfileResourceInfo, pName)

typedef struct KinetoPlugin_TraceBuilderHandle KinetoPlugin_TraceBuilderHandle;

// Trace Builder is an opaque object provided by Kineto that helps the plugin
// log event data without having to serialize down the wire over a C interface
// Trace Builder also takes care of memory management / ownership
struct KinetoPlugin_TraceBuilder {
  // Always set to KINETO_PLUGIN_TRACE_BUILDER_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle;


  // Usage: For each event, call addEvent(), then setLastEventName().
  // Optionally call addLastEventMetadata() and setLastEventFlow().
  // Use addDeviceInfo() and addResourceInfo() for device/resource metadata.
  //
  // Example:
  //   KinetoPlugin_ProfileEvent event = {
  //     .unpaddedStructSize = KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE,
  //     .eventType = KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME,
  //     .startTimeUtcNs = 1000000000, .endTimeUtcNs = 1000005000,
  //     .eventId = 1, .deviceId = 0, .resourceId = 123
  //   };
  //   traceBuilder->addEvent(traceBuilder->pTraceBuilderHandle, &event);
  //   traceBuilder->setLastEventName(traceBuilder->pTraceBuilderHandle,
  //   "cudaLaunchKernel");

  // Several of these APIs do not have failure cases, but we still return an
  // integer for future extensibility.

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int (*addEvent)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const struct KinetoPlugin_ProfileEvent* pProfileEvent);

  // returns 0 on success, -1 on failure, this can happen if the last event is not set.
  int (*setLastEventName)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const char* pName);

  // returns 0 on success, -1 on failure, this can happen if the last event is not set.
  int (*setLastEventFlow)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const struct KinetoPlugin_ProfileEventFlow* pProfileEventFlow);
  // If metadata value is a string itself (e.g., "metadata value"), MUST add
  // additional quote around it (e.g., "\"metadata value\"") If metadata value
  // is a list (e.g., 1, 2, 3), MUST add square bracket around it (e.g., "[1, 2,
  // 3]")
  // returns 0 on success, -1 on failure, this can happen if the last event is not set.
  int (*addLastEventMetadata)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const char* pKey,
      const char* pValue);

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int (*addDeviceInfo)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const struct KinetoPlugin_ProfileDeviceInfo* pProfileDeviceInfo);

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int (*addResourceInfo)(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const struct KinetoPlugin_ProfileResourceInfo* pProfileResourceInfo);
};
#define KINETO_PLUGIN_TRACE_BUILDER_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                    \
      struct KinetoPlugin_TraceBuilder, addResourceInfo)

typedef struct KinetoPlugin_ProfilerHandle KinetoPlugin_ProfilerHandle;

struct KinetoPlugin_ProfilerCreate_Params {
  // Always set to KINETO_PLUGIN_PROFILER_CREATE_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [out] An instance created by plugin
  KinetoPlugin_ProfilerHandle* pProfilerHandle;

  // [in] Enabled activity types.
  KinetoPlugin_ProfileEventType* pEnabledActivityTypes;

  // [in] Max length of pEnabledActivityTypes
  size_t enabledActivityTypesMaxLen;
};
#define KINETO_PLUGIN_PROFILER_CREATE_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                             \
      struct KinetoPlugin_ProfilerCreate_Params, enabledActivityTypesMaxLen)

struct KinetoPlugin_ProfilerDestroy_Params {
  // Always set to KINETO_PLUGIN_PROFILER_DESTROY_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_DESTROY_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                              \
      struct KinetoPlugin_ProfilerDestroy_Params, pProfilerHandle)

struct KinetoPlugin_ProfilerQuery_Params {
  // Always set to KINETO_PLUGIN_PROFILER_QUERY_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;

  // [in/out] Memory space to receive profiler name
  char* pProfilerName;

  // [in] Max length of pProfilerName excluding null terminator
  size_t profilerNameMaxLen;

  // [in/out] Supported activity types.
  KinetoPlugin_ProfileEventType* pSupportedActivityTypes;

  // [in] Max length of pSupportedActivityTypes
  size_t supportedActivityTypesMaxLen;
};
#define KINETO_PLUGIN_PROFILER_QUERY_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                            \
      struct KinetoPlugin_ProfilerQuery_Params, supportedActivityTypesMaxLen)

struct KinetoPlugin_ProfilerStart_Params {
  // Always set to KINETO_PLUGIN_PROFILER_START_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_START_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                            \
      struct KinetoPlugin_ProfilerStart_Params, pProfilerHandle)

struct KinetoPlugin_ProfilerStop_Params {
  // Always set to KINETO_PLUGIN_PROFILER_STOP_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_STOP_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                           \
      struct KinetoPlugin_ProfilerStop_Params, pProfilerHandle)

struct KinetoPlugin_ProfilerProcessEvents_Params {
  // Always set to
  // KINETO_PLUGIN_PROFILER_PROCESS_EVENTS_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;

  // [in] APIs to build a trace
  const struct KinetoPlugin_TraceBuilder* pTraceBuilder;
};
#define KINETO_PLUGIN_PROFILER_PROCESS_EVENTS_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                                     \
      struct KinetoPlugin_ProfilerProcessEvents_Params, pTraceBuilder)

struct KinetoPlugin_ProfilerPushCorrelationId_Params {
  // Always set to
  // KINETO_PLUGIN_PROFILER_PUSH_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] Correlation ID
  uint64_t correlationId;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_PUSH_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                                          \
      struct KinetoPlugin_ProfilerPushCorrelationId_Params, pProfilerHandle)

struct KinetoPlugin_ProfilerPopCorrelationId_Params {
  // Always set to
  // KINETO_PLUGIN_PROFILER_POP_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_POP_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                                         \
      struct KinetoPlugin_ProfilerPopCorrelationId_Params, pProfilerHandle)

struct KinetoPlugin_ProfilerPushUserCorrelationId_Params {
  // Always set to
  // KINETO_PLUGIN_PROFILER_PUSH_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] User correlation ID
  uint64_t userCorrelationId;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_PUSH_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                                               \
      struct KinetoPlugin_ProfilerPushUserCorrelationId_Params,                     \
      pProfilerHandle)

struct KinetoPlugin_ProfilerPopUserCorrelationId_Params {
  // Always set to
  // KINETO_PLUGIN_PROFILER_POP_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // [in] An instance created via profilerCreate()
  KinetoPlugin_ProfilerHandle* pProfilerHandle;
};
#define KINETO_PLUGIN_PROFILER_POP_USER_CORRELATION_ID_PARAMS_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                                              \
      struct KinetoPlugin_ProfilerPopUserCorrelationId_Params,                     \
      pProfilerHandle)

struct KinetoPlugin_ProfilerInterface {
  // Always set to KINETO_PLUGIN_PROFILER_INTERFACE_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  // Create an instance of the profiler session
  int (*profilerCreate)(
      struct KinetoPlugin_ProfilerCreate_Params* pProfilerCreateParams);

  // Destroy an instance of the profiler session
  int (*profilerDestroy)(
      struct KinetoPlugin_ProfilerDestroy_Params* pProfilerDestroyParams);

  // Query additional information about the profile such as its name, supported
  // activity types, etc.
  int (*profilerQuery)(
      struct KinetoPlugin_ProfilerQuery_Params* pProfilerQueryParams);

  // Start the trace collection
  int (*profilerStart)(
      struct KinetoPlugin_ProfilerStart_Params* pProfilerStartParams);

  // Stop the trace collection
  int (*profilerStop)(
      struct KinetoPlugin_ProfilerStop_Params* pProfilerStopParams);

  // The following interfaces are optional and can be used to track correlation
  // IDs If not implemented, the correlation IDs will not be tracked through the
  // profiler session.

  // Push a correlation ID to the profiler session
  int (*profilerPushCorrelationId)(
      struct KinetoPlugin_ProfilerPushCorrelationId_Params*
          pProfilerPushCorrelationIdParams);

  // Pop a correlation ID from the profiler session
  int (*profilerPopCorrelationId)(
      struct KinetoPlugin_ProfilerPopCorrelationId_Params*
          pProfilerPopCorrelationIdParams);

  // Push a user correlation ID to the profiler session
  int (*profilerPushUserCorrelationId)(
      struct KinetoPlugin_ProfilerPushUserCorrelationId_Params*
          pProfilerPushUserCorrelationIdParams);

  // Pop a user correlation ID from the profiler session
  int (*profilerPopUserCorrelationId)(
      struct KinetoPlugin_ProfilerPopUserCorrelationId_Params*
          pProfilerPopUserCorrelationIdParams);

  // Process the events collected by the profiler session to the central
  // profiler. This leverages the trace builder to enqueue events.
  int (*profilerProcessEvents)(struct KinetoPlugin_ProfilerProcessEvents_Params*
                                   pProfilerProcessEventsParams);
};
#define KINETO_PLUGIN_PROFILER_INTERFACE_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(                         \
      struct KinetoPlugin_ProfilerInterface, profilerProcessEvents)

typedef struct KinetoPlugin_RegistryHandle KinetoPlugin_RegistryHandle;

struct KinetoPlugin_Registry {
  // Always set to KINETO_PLUGIN_REGISTRY_UNPADDED_STRUCT_SIZE
  size_t unpaddedStructSize;

  KinetoPlugin_RegistryHandle* pRegistryHandle;

  int (*registerProfiler)(
      KinetoPlugin_RegistryHandle* pRegistryHandle,
      const struct KinetoPlugin_ProfilerInterface* pProfiler);
};
#define KINETO_PLUGIN_REGISTRY_UNPADDED_STRUCT_SIZE \
  KINETO_PLUGIN_UNPADDED_STRUCT_SIZE(               \
      struct KinetoPlugin_Registry, registerProfiler)

// .so function signature
// To be implemented by plugin
// To be called by Kineto
int KinetoPlugin_register(const struct KinetoPlugin_Registry* pRegistry);

#ifdef __cplusplus
}
#endif

#endif // KINETO_DYNAMIC_PLUGIN_FORMAT_H
