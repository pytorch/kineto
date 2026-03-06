/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityType.h"

#include <fmt/format.h>

namespace libkineto {

struct ActivityTypeName {
  const char* name;
  ActivityType type;
};

static constexpr std::array<ActivityTypeName, activityTypeCount + 1> map{
    {{.name = "cpu_op", .type = ActivityType::CPU_OP},
     {.name = "user_annotation", .type = ActivityType::USER_ANNOTATION},
     {.name = "gpu_user_annotation", .type = ActivityType::GPU_USER_ANNOTATION},
     {.name = "gpu_memcpy", .type = ActivityType::GPU_MEMCPY},
     {.name = "gpu_memset", .type = ActivityType::GPU_MEMSET},
     {.name = "kernel", .type = ActivityType::CONCURRENT_KERNEL},
     {.name = "external_correlation",
      .type = ActivityType::EXTERNAL_CORRELATION},
     {.name = "cuda_runtime", .type = ActivityType::CUDA_RUNTIME},
     {.name = "cuda_driver", .type = ActivityType::CUDA_DRIVER},
     {.name = "cpu_instant_event", .type = ActivityType::CPU_INSTANT_EVENT},
     {.name = "python_function", .type = ActivityType::PYTHON_FUNCTION},
     {.name = "overhead", .type = ActivityType::OVERHEAD},
     {.name = "mtia_runtime", .type = ActivityType::MTIA_RUNTIME},
     {.name = "mtia_ccp_events", .type = ActivityType::MTIA_CCP_EVENTS},
     {.name = "mtia_insight", .type = ActivityType::MTIA_INSIGHT},
     {.name = "cuda_sync", .type = ActivityType::CUDA_SYNC},
     {.name = "cuda_event", .type = ActivityType::CUDA_EVENT},
     {.name = "glow_runtime", .type = ActivityType::GLOW_RUNTIME},
     {.name = "cuda_profiler_range", .type = ActivityType::CUDA_PROFILER_RANGE},
     {.name = "hpu_op", .type = ActivityType::HPU_OP},
     {.name = "xpu_runtime", .type = ActivityType::XPU_RUNTIME},
     {.name = "xpu_driver", .type = ActivityType::XPU_DRIVER},
     {.name = "collective_comm", .type = ActivityType::COLLECTIVE_COMM},
     {.name = "privateuse1_runtime", .type = ActivityType::PRIVATEUSE1_RUNTIME},
     {.name = "privateuse1_driver", .type = ActivityType::PRIVATEUSE1_DRIVER},
     {.name = "ENUM_COUNT", .type = ActivityType::ENUM_COUNT}}};

static constexpr bool matchingOrder(int idx = 0) {
  return map[idx].type == ActivityType::ENUM_COUNT ||
      ((idx == (int)map[idx].type) && matchingOrder(idx + 1));
}
static_assert(matchingOrder(), "ActivityTypeName map is out of order");

const char* toString(ActivityType t) {
  return map[(int)t].name;
}

ActivityType toActivityType(const std::string& str) {
  for (int i = 0; i < activityTypeCount; i++) {
    if (str == map[i].name) {
      return map[i].type;
    }
  }
  throw std::invalid_argument(fmt::format("Invalid activity type: {}", str));
}

std::array<ActivityType, activityTypeCount> activityTypes() {
  std::array<ActivityType, activityTypeCount> res;
  for (int i = 0; i < activityTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}

std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes() {
  std::array<ActivityType, defaultActivityTypeCount> res;
  for (int i = 0; i < defaultActivityTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}

} // namespace libkineto
