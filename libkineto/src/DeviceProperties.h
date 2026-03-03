/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#ifdef HAS_CUPTI
#include <cupti.h>
#endif

namespace KINETO_NAMESPACE {

// Return compute properties for each device as a json string
const std::string& devicePropertiesJson();

int smCount(uint32_t deviceId);

// TODO: Implement the below for HAS_ROCTRACER
#ifdef HAS_CUPTI

// Use newer CUPTI activity structs in CUDA 12.0+ for extended fields
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
using CUpti_ActivityKernelType = CUpti_ActivityKernel9;
using CUpti_ActivityMemcpyType = CUpti_ActivityMemcpy5;
using CUpti_ActivityMemcpyPtoPType = CUpti_ActivityMemcpyPtoP4;
using CUpti_ActivityMemsetType = CUpti_ActivityMemset4;
#else
using CUpti_ActivityKernelType = CUpti_ActivityKernel4;
using CUpti_ActivityMemcpyType = CUpti_ActivityMemcpy;
using CUpti_ActivityMemcpyPtoPType = CUpti_ActivityMemcpy2;
using CUpti_ActivityMemsetType = CUpti_ActivityMemset;
#endif

float blocksPerSm(const CUpti_ActivityKernelType& kernel);
float warpsPerSm(const CUpti_ActivityKernelType& kernel);

// Return estimated achieved occupancy for a kernel
float kernelOccupancy(const CUpti_ActivityKernelType& kernel);
float kernelOccupancy(uint32_t deviceId,
                      uint16_t registersPerThread,
                      int32_t staticSharedMemory,
                      int32_t dynamicSharedMemory,
                      int32_t blockX,
                      int32_t blockY,
                      int32_t blockZ,
                      float blocks_per_sm);
#endif

} // namespace KINETO_NAMESPACE
