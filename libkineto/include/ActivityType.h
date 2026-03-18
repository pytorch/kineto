/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <set>
#include <string>

namespace libkineto {

// Note : All activity types are not enabled by default. Please add them
// at correct position in the enum
enum class ActivityType {
  // Activity types enabled by default
  CPU_OP = 0, // cpu side ops
  USER_ANNOTATION = 1,
  GPU_USER_ANNOTATION = 2,
  GPU_MEMCPY = 3,
  GPU_MEMSET = 4,
  CONCURRENT_KERNEL = 5, // on-device kernels
  EXTERNAL_CORRELATION = 6,
  CUDA_RUNTIME = 7, // host side cuda runtime events
  CUDA_DRIVER = 8, // host side cuda driver events
  CPU_INSTANT_EVENT = 9, // host side point-like events
  PYTHON_FUNCTION = 10,
  OVERHEAD = 11, // CUPTI induced overhead events sampled from its overhead API.
  MTIA_RUNTIME = 12, // host side MTIA runtime events
  MTIA_CCP_EVENTS = 13, // MTIA ondevice CCP events
  MTIA_INSIGHT = 14, // MTIA Insight Events
  CUDA_SYNC = 15, // synchronization events between runtime and kernels
  CUDA_EVENT = 16, // CUDA event activities (cudaEventRecord, etc.)

  // Optional Activity types
  GLOW_RUNTIME = 17, // host side glow runtime events
  CUDA_PROFILER_RANGE = 18, // CUPTI Profiler range for performance metrics
  HPU_OP = 19, // HPU host side runtime event
  XPU_RUNTIME = 20, // host side xpu runtime events
  XPU_DRIVER = 21, // host side xpu driver events
  XPU_SCOPE_PROFILER = 22, // XPUPTI Profiler scope for performance metrics
  COLLECTIVE_COMM = 23, // collective communication

  // PRIVATEUSE1 Activity types are used for custom backends.
  // The corresponding device type is `DeviceType::PrivateUse1` in PyTorch.
  PRIVATEUSE1_RUNTIME = 24, // host side privateUse1 runtime events
  PRIVATEUSE1_DRIVER = 25, // host side privateUse1 driver events

  ENUM_COUNT =
      26, // This is to add buffer and not used for any profiling logic. Add
  // your new type before it.
  OPTIONAL_ACTIVITY_TYPE_START = GLOW_RUNTIME,
};

const char* toString(ActivityType t);
ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
constexpr int defaultActivityTypeCount =
    (int)ActivityType::OPTIONAL_ACTIVITY_TYPE_START;
std::array<ActivityType, activityTypeCount> activityTypes();
std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes();

} // namespace libkineto
