/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kineto/libkineto/sample_programs/kineto_rocm_stream_interleave_repro_kernels.h"

#include <hip/hip_runtime.h>

namespace kineto::samples {
namespace {

template <int WorkerId>
__global__ void kineto_issue_c_worker_kernel(
    float* buffer,
    int elements,
    int iterations) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int offset = idx; offset < elements; offset += stride) {
    float value = buffer[offset] + static_cast<float>(WorkerId + 1);
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
      value = value * 1.000001f +
          static_cast<float>((WorkerId + 1) * 0.000001f);
    }
    buffer[offset] = value;
  }
}

template <int WorkerId>
hipError_t launchTypedWorkerKernel(
    float* buffer,
    int elements,
    int iterations,
    hipStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  const int blocks = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  hipLaunchKernelGGL(
      (kineto_issue_c_worker_kernel<WorkerId>),
      dim3(blocks),
      dim3(kThreadsPerBlock),
      0,
      stream,
      buffer,
      elements,
      iterations);
  return hipGetLastError();
}

} // namespace

hipError_t launchIssueCWorkerKernel(
    int workerId,
    float* buffer,
    int elements,
    int iterations,
    hipStream_t stream) {
  switch (workerId) {
    case 0:
      return launchTypedWorkerKernel<0>(buffer, elements, iterations, stream);
    case 1:
      return launchTypedWorkerKernel<1>(buffer, elements, iterations, stream);
    case 2:
      return launchTypedWorkerKernel<2>(buffer, elements, iterations, stream);
    case 3:
      return launchTypedWorkerKernel<3>(buffer, elements, iterations, stream);
    case 4:
      return launchTypedWorkerKernel<4>(buffer, elements, iterations, stream);
    case 5:
      return launchTypedWorkerKernel<5>(buffer, elements, iterations, stream);
    case 6:
      return launchTypedWorkerKernel<6>(buffer, elements, iterations, stream);
    case 7:
      return launchTypedWorkerKernel<7>(buffer, elements, iterations, stream);
    case 8:
      return launchTypedWorkerKernel<8>(buffer, elements, iterations, stream);
    case 9:
      return launchTypedWorkerKernel<9>(buffer, elements, iterations, stream);
    case 10:
      return launchTypedWorkerKernel<10>(buffer, elements, iterations, stream);
    case 11:
      return launchTypedWorkerKernel<11>(buffer, elements, iterations, stream);
    case 12:
      return launchTypedWorkerKernel<12>(buffer, elements, iterations, stream);
    case 13:
      return launchTypedWorkerKernel<13>(buffer, elements, iterations, stream);
    case 14:
      return launchTypedWorkerKernel<14>(buffer, elements, iterations, stream);
    case 15:
      return launchTypedWorkerKernel<15>(buffer, elements, iterations, stream);
    default:
      return hipErrorInvalidValue;
  }
}

} // namespace kineto::samples
