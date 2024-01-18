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

static constexpr std::array<ActivityTypeName, activityTypeCount + 1> map{{
    {"cpu_op", ActivityType::CPU_OP},
    {"user_annotation", ActivityType::USER_ANNOTATION},
    {"gpu_user_annotation", ActivityType::GPU_USER_ANNOTATION},
    {"gpu_memcpy", ActivityType::GPU_MEMCPY},
    {"gpu_memset", ActivityType::GPU_MEMSET},
    {"kernel", ActivityType::CONCURRENT_KERNEL},
    {"external_correlation", ActivityType::EXTERNAL_CORRELATION},
    {"cuda_runtime", ActivityType::CUDA_RUNTIME},
    {"cuda_driver", ActivityType::CUDA_DRIVER},
    {"cpu_instant_event", ActivityType::CPU_INSTANT_EVENT},
    {"python_function", ActivityType::PYTHON_FUNCTION},
    {"overhead", ActivityType::OVERHEAD},
    {"mtia_runtime", ActivityType::MTIA_RUNTIME},
    {"mtia_ccp_events", ActivityType::MTIA_CCP_EVENTS},
    {"cuda_sync", ActivityType::CUDA_SYNC},
    {"glow_runtime", ActivityType::GLOW_RUNTIME},
    {"cuda_profiler_range", ActivityType::CUDA_PROFILER_RANGE},
    {"hpu_op", ActivityType::HPU_OP},
    {"xpu_runtime", ActivityType::XPU_RUNTIME},
    {"collective_comm", ActivityType::COLLECTIVE_COMM},
    {"ENUM_COUNT", ActivityType::ENUM_COUNT}
}};

static constexpr bool matchingOrder(int idx = 0) {
  return map[idx].type == ActivityType::ENUM_COUNT ||
    ((idx == (int) map[idx].type) && matchingOrder(idx + 1));
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

const std::array<ActivityType, activityTypeCount> activityTypes() {
  std::array<ActivityType, activityTypeCount> res;
  for (int i = 0; i < activityTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}

const std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes() {
  std::array<ActivityType, defaultActivityTypeCount> res;
  for (int i = 0; i < defaultActivityTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}


} // namespace libkineto
