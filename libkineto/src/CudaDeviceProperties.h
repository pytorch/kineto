/*
 * Copyright (c) Kineto Contributors
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <stdint.h>
#include <cuda_occupancy.h>

namespace KINETO_NAMESPACE {

const std::vector<cudaOccDeviceProp>& occDeviceProps();

float kernelOccupancy(
    uint32_t deviceId,
    uint16_t registersPerThread,
    int32_t staticSharedMemory,
    int32_t dynamicSharedMemory,
    int32_t blockX,
    int32_t blockY,
    int32_t blockZ,
    float blocks_per_sm);

} // namespace KINETO_NAMESPACE
