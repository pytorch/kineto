#pragma once

#include "ActivityType.h"
#include "KinetoDynamicPluginInterface.h"
#include "GenericTraceActivity.h"

namespace libkineto {

// Utility functions for dynamic plugins

static inline ActivityType
convertToActivityType(KinetoPlugin_ProfileEventType type) {
  switch (type) {
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_OP:
    return ActivityType::CPU_OP;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_USER_ANNOTATION:
    return ActivityType::USER_ANNOTATION;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_USER_ANNOTATION:
    return ActivityType::GPU_USER_ANNOTATION;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMCPY:
    return ActivityType::GPU_MEMCPY;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMSET:
    return ActivityType::GPU_MEMSET;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL:
    return ActivityType::CONCURRENT_KERNEL;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_EXTERNAL_CORRELATION:
    return ActivityType::EXTERNAL_CORRELATION;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME:
    return ActivityType::CUDA_RUNTIME;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_DRIVER:
    return ActivityType::CUDA_DRIVER;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_INSTANT_EVENT:
    return ActivityType::CPU_INSTANT_EVENT;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_PYTHON_FUNCTION:
    return ActivityType::PYTHON_FUNCTION;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_OVERHEAD:
    return ActivityType::OVERHEAD;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_SYNC:
    return ActivityType::CUDA_SYNC;
  case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_PM_COUNTER:
    return ActivityType::GPU_PM_COUNTER;
  default:
    // use kernel type as a default
    return ActivityType::CONCURRENT_KERNEL;
  }
}

static inline unsigned int
convertToLinkType(KinetoPlugin_ProfileEventFlowType type) {
  switch (type) {
  case KINETO_PLUGIN_PROFILE_EVENT_FLOW_TYPE_FWD_BWD:
    return kLinkFwdBwd;
  case KINETO_PLUGIN_PROFILE_EVENT_FLOW_TYPE_ASYNC_CPU_GPU:
    return kLinkAsyncCpuGpu;
  default:
    return 0;
  }
}

} // namespace libkineto
