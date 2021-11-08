/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    GLOW_RUNTIME, // host side glow runtime events
    CPU_INSTANT_EVENT, // host side point-like events
    PYTHON_FUNCTION,
    ENUM_COUNT
};

const char* toString(ActivityType t);
ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
const std::array<ActivityType, activityTypeCount> activityTypes();

} // namespace libkineto
