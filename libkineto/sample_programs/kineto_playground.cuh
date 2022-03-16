// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>

namespace kineto {

// Warms up CUDA before the tracing starts
void warmup(void);

// Basic usage of cudaMemcpy and cudaMemset
void basicMemcpyToDevice(void);

// Basic usage of cudaMemcpy from device to host
void basicMemcpyFromDevice(void);

// Your experimental code goes in here!
void playground(void);

// Run a simple elementwise kernel
void compute(void);

}
