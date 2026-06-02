/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime_api.h>

namespace kineto::samples {

constexpr int kIssueCMaxWorkers = 16;

hipError_t launchIssueCWorkerKernel(int workerId, float* buffer, int elements, int iterations, hipStream_t stream);

} // namespace kineto::samples
