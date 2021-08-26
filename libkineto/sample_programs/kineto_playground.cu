// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <stdio.h>

#include "kineto_playground.cuh"


namespace kineto {

void warmup(void) {
  // Inititalizing CUDA can take a while which we normally do not want to see in Kineto traces.
  // This is done in various ways that take Kineto as dependency. This is our way of doing warmup
  // for kineto_playground
	size_t bytes = 1000;
	float* mem = NULL;
	auto error = cudaMalloc(&mem, bytes);
  if (error != cudaSuccess) {
    printf("cudaMalloc failed during kineto_playground warmup. error code: %d", error);
    return;
  }

  cudaFree(mem); 
}

void basicMemcpyMemset(void) {
  size_t size = (1 << 8) * sizeof(float);
  float *hostMemSrc, *deviceMem, *hostMemDst;
  cudaError_t err;

  hostMemSrc = (float*)malloc(size);
  hostMemDst = (float*)malloc(size);
  err = cudaMalloc(&deviceMem, size);
  if (err != cudaSuccess) {
    printf("cudaMalloc failed during %s", __func__);
    return;
  }

  memset(hostMemSrc, 1, size);
  cudaMemcpy(deviceMem, hostMemSrc, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("cudaMemcpy failed during %s", __func__);
    return;
  }

  cudaMemcpy(hostMemDst, deviceMem, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy failed during %s", __func__);
    return;
  }

  free(hostMemSrc);
  free(hostMemDst);
  cudaFree(deviceMem);
}

void playground(void) {
  // Add your experimental CUDA implementation here. 
}

}
