/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <string>

#include <cupti.h>

namespace KINETO_NAMESPACE {

int smCount(uint32_t deviceId);
float blocksPerSm(const CUpti_ActivityKernel4& kernel);
float warpsPerSm(const CUpti_ActivityKernel4& kernel);

// Return estimated achieved occupancy for a kernel
float kernelOccupancy(const CUpti_ActivityKernel4& kernel);
float kernelOccupancy(
    uint32_t deviceId,
    uint16_t registersPerThread,
    int32_t staticSharedMemory,
    int32_t dynamicSharedMemory,
    int32_t blockX,
    int32_t blockY,
    int32_t blockZ,
    float blocks_per_sm);

// Return compute properties for each device as a json string
const std::string& devicePropertiesJson();

} // namespace KINETO_NAMESPACE
