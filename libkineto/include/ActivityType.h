// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <string>

namespace libkineto {

enum class ActivityType {
    CPU_OP = 0, // cpu side ops
    USER_ANNOTATION,
    GPU_USER_ANNOTATION,
    GPU_MEMCPY,
    GPU_MEMSET,
    CONCURRENT_KERNEL, // on-device kernels
    EXTERNAL_CORRELATION,
    CUDA_RUNTIME, // host side cuda runtime events
    CUDA_PROFILER_RANGE, // CUPTI Profiler range for performance metrics
    GLOW_RUNTIME, // host side glow runtime events
    CPU_INSTANT_EVENT, // host side point-like events
    PYTHON_FUNCTION,
    OVERHEAD, // CUPTI induced overhead events sampled from its overhead API.
    ENUM_COUNT // This is to add buffer and not used for any profiling logic. Add your new type before it.
};

const char* toString(ActivityType t);
ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
const std::array<ActivityType, activityTypeCount> activityTypes();

} // namespace libkineto
