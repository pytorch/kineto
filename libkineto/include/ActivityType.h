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

enum class ActivityType {
  // -------------------------------------------------------------------------
  // Accelerator-Agnostic Event Types
  // -------------------------------------------------------------------------
  // These are the canonical event types that work across all accelerators.
  // Prefer using these over device-specific types for better extensibility
  // and maintainability.

  CPU_OP = 0, // CPU-side ops (e.g., from PyTorch)
  USER_ANNOTATION, // User-defined annotations
  GPU_USER_ANNOTATION, // GPU-side user annotations
  GPU_MEMCPY, // Memory copy operations
  GPU_MEMSET, // Memory set operations
  CONCURRENT_KERNEL, // On-device kernel execution
  EXTERNAL_CORRELATION, // Correlation with external events
  RUNTIME, // Host-side runtime events
  DRIVER, // Host-side driver events
  CPU_INSTANT_EVENT, // Host-side point-like events
  PYTHON_FUNCTION, // Python function calls
  OVERHEAD, // Profiler-induced overhead events
  COLLECTIVE_COMM, // Collective communication operations
  GPU_PM_COUNTER, // Performance monitoring counters

  // -------------------------------------------------------------------------
  // Device-Specific Event Types
  // -------------------------------------------------------------------------
  // These events don't fit into the accelerator-agnostic categories above.
  // Use sparingly; prefer agnostic types when possible.

  MTIA_INSIGHT, // MTIA Insight events
  CUDA_SYNC, // CUDA synchronization events
  CUDA_EVENT, // CUDA event activities (cudaEventRecord, etc.)
  CUDA_PROFILER_RANGE, // CUPTI Profiler range for performance metrics

  // -------------------------------------------------------------------------
  ENUM_COUNT, // Sentinel value; add new types above this line

  // -------------------------------------------------------------------------
  // Aliased Event Types (Deprecated)
  // -------------------------------------------------------------------------
  // These are aliases to accelerator-agnostic types for backward compatibility.
  // Do NOT add new aliases. We aim to remove these in the future.

  CUDA_RUNTIME = RUNTIME,
  CUDA_DRIVER = DRIVER,
  MTIA_RUNTIME = RUNTIME,
  MTIA_CCP_EVENTS = CONCURRENT_KERNEL,
  GLOW_RUNTIME = RUNTIME,
  HPU_OP = RUNTIME,
  XPU_RUNTIME = RUNTIME,

  // PrivateUse1: Custom backend activity types
  // Corresponds to DeviceType::PrivateUse1 in PyTorch.
  PRIVATEUSE1_RUNTIME = RUNTIME,
  PRIVATEUSE1_DRIVER = DRIVER,
};

const char* toString(ActivityType t);
ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;

// Return an array of all activity types, note does not return aliases.
const std::array<ActivityType, activityTypeCount> activityTypes();

// Default activity types that are enabled by default during profiling
inline constexpr std::array defaultActivityTypesArray = {
    ActivityType::CPU_OP,
    ActivityType::USER_ANNOTATION,
    ActivityType::GPU_USER_ANNOTATION,
    ActivityType::GPU_MEMCPY,
    ActivityType::GPU_MEMSET,
    ActivityType::CONCURRENT_KERNEL,
    ActivityType::EXTERNAL_CORRELATION,
    ActivityType::RUNTIME,
    ActivityType::DRIVER,
    ActivityType::CPU_INSTANT_EVENT,
    ActivityType::PYTHON_FUNCTION,
    ActivityType::OVERHEAD,
    ActivityType::MTIA_RUNTIME,
    ActivityType::MTIA_CCP_EVENTS,
    ActivityType::MTIA_INSIGHT,
    ActivityType::CUDA_SYNC,
    ActivityType::CUDA_EVENT,
};

constexpr int defaultActivityTypeCount = defaultActivityTypesArray.size();

constexpr auto defaultActivityTypes() {
  return defaultActivityTypesArray;
}

} // namespace libkineto
