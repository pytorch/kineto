/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CudaUtil.h"

#ifdef HAS_CUPTI
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <mutex>

namespace KINETO_NAMESPACE {

bool gpuAvailable = false;

bool isGpuAvailable() {
#ifdef HAS_CUPTI
  static std::once_flag once;
  std::call_once(once, [] {
    // determine GPU availability on the system
    cudaError_t error;
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    gpuAvailable = (error == cudaSuccess && deviceCount > 0);
  });

  return gpuAvailable;
#else
  return false;
#endif
}

} // namespace KINETO_NAMESPACE
