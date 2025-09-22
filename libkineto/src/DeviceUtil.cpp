/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DeviceUtil.h"

namespace KINETO_NAMESPACE {

bool isAMDGpuAvailable() {
#ifdef HAS_ROCTRACER
  static bool amdGpuAvailable = [] {
    // determine AMD GPU availability on the system
    hipError_t error;
    int deviceCount;
    error = hipGetDeviceCount(&deviceCount);
    return (error == hipSuccess && deviceCount > 0);
  }();
  return amdGpuAvailable;
#else
  return false;
#endif
}

bool isCUDAGpuAvailable() {
#ifdef HAS_CUPTI
  static bool cudaGpuAvailable = [] {
    // determine CUDA GPU availability on the system
    cudaError_t error;
    int deviceCount = 0;
    error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
  }();
  return cudaGpuAvailable;
#else
  return false;
#endif
}

bool isGpuAvailable() {
  bool amd = isAMDGpuAvailable();
  bool cuda = isCUDAGpuAvailable();
  return amd || cuda;
}

} // namespace KINETO_NAMESPACE
