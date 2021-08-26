// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>

namespace kineto {

// Warms up CUDA before the tracing starts
void warmup(void);

// Basic usage of cudaMemcpy and cudaMemset
void basicMemcpyMemset(void);

// Your experimental code goes in here!
void playground(void);

}
