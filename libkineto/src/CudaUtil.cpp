// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CudaUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>

namespace KINETO_NAMESPACE {

bool gpuAvailable = false;

bool isGpuAvailable() {
  static std::once_flag once;
  std::call_once(once, [] {
    // determine GPU availability on the system
    cudaError_t error;
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    gpuAvailable = (error == cudaSuccess && deviceCount > 0);
  });

  return gpuAvailable;
}

} // namespace KINETO_NAMESPACE
