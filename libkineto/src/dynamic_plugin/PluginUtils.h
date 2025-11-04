#pragma once

#include <set>
#include <vector>

#include "ActivityType.h"
#include "GenericTraceActivity.h"
#include "KinetoDynamicPluginInterface.h"

namespace libkineto {

// Utility functions for dynamic plugins

static inline ActivityType convertToActivityType(
    KinetoPlugin_ProfileEventType type) {
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
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_RUNTIME:
      return ActivityType::MTIA_RUNTIME;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_CCP_EVENTS:
      return ActivityType::MTIA_CCP_EVENTS;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_INSIGHT:
      return ActivityType::MTIA_INSIGHT;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_SYNC:
      return ActivityType::CUDA_SYNC;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GLOW_RUNTIME:
      return ActivityType::GLOW_RUNTIME;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_PROFILER_RANGE:
      return ActivityType::CUDA_PROFILER_RANGE;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_HPU_OP:
      return ActivityType::HPU_OP;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_XPU_RUNTIME:
      return ActivityType::XPU_RUNTIME;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_COLLECTIVE_COMM:
      return ActivityType::COLLECTIVE_COMM;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_PM_COUNTER:
      return ActivityType::GPU_PM_COUNTER;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_RUNTIME:
      return ActivityType::PRIVATEUSE1_RUNTIME;
    case KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_DRIVER:
      return ActivityType::PRIVATEUSE1_DRIVER;
    default:
      // use kernel type as a default
      return ActivityType::CONCURRENT_KERNEL;
  }
}

static inline KinetoPlugin_ProfileEventType convertFromActivityType(
    ActivityType type) {
  switch (type) {
    case ActivityType::CPU_OP:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_OP;
    case ActivityType::USER_ANNOTATION:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_USER_ANNOTATION;
    case ActivityType::GPU_USER_ANNOTATION:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_USER_ANNOTATION;
    case ActivityType::GPU_MEMCPY:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMCPY;
    case ActivityType::GPU_MEMSET:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMSET;
    case ActivityType::CONCURRENT_KERNEL:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL;
    case ActivityType::EXTERNAL_CORRELATION:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_EXTERNAL_CORRELATION;
    case ActivityType::CUDA_RUNTIME:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME;
    case ActivityType::CUDA_DRIVER:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_DRIVER;
    case ActivityType::CPU_INSTANT_EVENT:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CPU_INSTANT_EVENT;
    case ActivityType::PYTHON_FUNCTION:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_PYTHON_FUNCTION;
    case ActivityType::OVERHEAD:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_OVERHEAD;
    case ActivityType::MTIA_RUNTIME:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_RUNTIME;
    case ActivityType::MTIA_CCP_EVENTS:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_CCP_EVENTS;
    case ActivityType::MTIA_INSIGHT:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_MTIA_INSIGHT;
    case ActivityType::CUDA_SYNC:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_SYNC;
    case ActivityType::GLOW_RUNTIME:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_GLOW_RUNTIME;
    case ActivityType::CUDA_PROFILER_RANGE:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_PROFILER_RANGE;
    case ActivityType::HPU_OP:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_HPU_OP;
    case ActivityType::XPU_RUNTIME:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_XPU_RUNTIME;
    case ActivityType::COLLECTIVE_COMM:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_COLLECTIVE_COMM;
    case ActivityType::GPU_PM_COUNTER:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_PM_COUNTER;
    case ActivityType::PRIVATEUSE1_RUNTIME:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_RUNTIME;
    case ActivityType::PRIVATEUSE1_DRIVER:
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_PRIVATEUSE1_DRIVER;
    default:
      // use kernel type as a default
      return KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL;
  }
}

static inline std::vector<KinetoPlugin_ProfileEventType> convertActivityTypeSet(
    const std::set<ActivityType>& activityTypes) {
  std::vector<KinetoPlugin_ProfileEventType> result;
  result.reserve(activityTypes.size());
  for (const auto& activityType : activityTypes) {
    result.push_back(convertFromActivityType(activityType));
  }
  return result;
}

static inline unsigned int convertToLinkType(
    KinetoPlugin_ProfileEventFlowType type) {
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
