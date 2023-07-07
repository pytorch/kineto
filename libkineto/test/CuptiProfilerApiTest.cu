/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cuda.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "src/Logger.h"
#include "src/CuptiRangeProfilerApi.h"

#define DRIVER_API_CALL(apiFuncCall)                           \
  do {                                                         \
    CUresult _status = apiFuncCall;                            \
    if (_status != CUDA_SUCCESS) {                             \
      LOG(ERROR) << "Failed invoking CUDA driver function "    \
                 << #apiFuncCall << " status = "               \
                 << _status;                                   \
      exit(-1);                                                \
    }                                                          \
  } while (0)

#define EXPECT(expr)\
  if (!(expr)) {\
  };

using namespace KINETO_NAMESPACE;

static int numRanges = 1;

using Type = double;

// Device code
__global__ void VecAdd(const Type* A, const Type* B, Type* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// Device code
__global__ void VecSub(const Type* A, const Type* B, Type* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] - B[i];
  }
}

static void initVec(Type* vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = i;
  }
}

static void cleanUp(
    Type* h_A,
    Type* h_B,
    Type* h_C,
    Type* h_D,
    Type* d_A,
    Type* d_B,
    Type* d_C,
    Type* d_D) {
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);
  if (d_D)
    cudaFree(d_D);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
  if (h_D)
    free(h_D);
}

/* Benchmark application used to test profiler measurements
 * This simply runs two kernels vector Add and Vector Subtract
 */

void VectorAddSubtract() {
  int N = 50000;
  size_t size = N * sizeof(Type);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  Type *h_A, *h_B, *h_C, *h_D;
  Type *d_A, *d_B, *d_C, *d_D;
  int i;
  Type sum, diff;

  // Allocate input vectors h_A and h_B in host memory
  h_A = (Type*)malloc(size);
  h_B = (Type*)malloc(size);
  h_C = (Type*)malloc(size);
  h_D = (Type*)malloc(size);

  // Initialize input vectors
  initVec(h_A, N);
  initVec(h_B, N);
  memset(h_C, 0, size);
  memset(h_D, 0, size);

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);
  cudaMalloc((void**)&d_D, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  threadsPerBlock = 256;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  LOG(INFO) << fmt::format(
      "Launching kernel: blocks {}, thread/block {}",
      blocksPerGrid,
      threadsPerBlock);

  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (i = 0; i < N; ++i) {
    sum = h_A[i] + h_B[i];
    diff = h_A[i] - h_B[i];
    if (h_C[i] != sum || h_D[i] != diff) {
      LOG(ERROR) << "Result verification failed";
      break;
    }
  }

  cleanUp(h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D);
}

#if HAS_CUPTI_RANGE_PROFILER
bool runTestWithAutoRange(
    int deviceNum,
    const std::vector<std::string>& metricNames,
    CUcontext cuContext,
    bool async) {

  // create a CUPTI range based profiling profiler
  //  this configures the counter data as well
  CuptiRangeProfilerOptions opts{
    .metricNames = metricNames,
    .deviceId = deviceNum,
    .maxRanges = 2,
    .numNestingLevels = 1,
    .cuContext = async ? nullptr : cuContext
  };
  CuptiRBProfilerSession profiler(opts);

  CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;

  if (async) {
    profiler.asyncStartAndEnable(profilerRange, profilerReplayMode);
  } else {
    profiler.start(profilerRange, profilerReplayMode);
    profiler.enable();
  }

  VectorAddSubtract();

  if (!async) {
    profiler.disable();
    // stop profiler
    profiler.stop();
  } else {
    profiler.asyncDisableAndStop();
  }

  auto result = profiler.evaluateMetrics(true);

  // check results
  EXPECT_EQ(result.metricNames.size(), 3);
  EXPECT_EQ(result.rangeVals.size(), 2);

  for (const auto& measurement : result.rangeVals) {
    EXPECT_EQ(measurement.values.size(), 3);

    if (measurement.values.size() == 3) {
      // smsp__warps_launched.avg
      EXPECT_NE(measurement.values[0], 0);
      // smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
      // each kernel has 50000 dadd ops
      EXPECT_EQ(measurement.values[1], 50000);
      // sm__inst_executed_pipe_tensor.sum
      //EXPECT_EQ(measurement.values[2], 0);
    }
  }
  return true;
}

bool runTestWithUserRange(
    int deviceNum,
    const std::vector<std::string>& metricNames,
    CUcontext cuContext,
    bool async = false) {

  // create a CUPTI range based profiling profiler
  //  this configures the counter data as well
  CuptiRangeProfilerOptions opts{
    .metricNames = metricNames,
    .deviceId = deviceNum,
    .maxRanges = numRanges,
    .numNestingLevels = 1,
    .cuContext = async ? nullptr : cuContext
  };
  CuptiRBProfilerSession profiler(opts);

  CUpti_ProfilerRange profilerRange = CUPTI_UserRange;
  CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;

  if (async) {
    profiler.asyncStartAndEnable(profilerRange, profilerReplayMode);
    { VectorAddSubtract(); }
    profiler.disableAndStop();
  } else {
    profiler.start(profilerRange, profilerReplayMode);

    /* User takes the resposiblity of replaying the kernel launches */
    bool replay = true;
    do {
      profiler.beginPass();
      {
        profiler.enable();

        std::string rangeName = "vecAddSub";
        profiler.pushRange(rangeName);

        { VectorAddSubtract(); }

        profiler.popRange();
        profiler.disable();
      }
      LOG(INFO) << "Replay starting.";
      replay = profiler.endPass();

    } while (!replay);

    // stop profiler
    profiler.stop();
  }
  VectorAddSubtract();
  auto result = profiler.evaluateMetrics(true);

  // check results
  EXPECT_EQ(result.metricNames.size(), 3);
  EXPECT_EQ(result.rangeVals.size(), 1);

  if (result.rangeVals.size() > 0) {
    const auto& measurement = result.rangeVals[0];
    EXPECT_EQ(measurement.values.size(), 3);

    if (measurement.values.size() == 3) {
      // smsp__warps_launched.avg
      EXPECT_NE(measurement.values[0], 0);
      // smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
      // in async mode multiple passes are not supported yet
      if (!async) {
        EXPECT_EQ(measurement.values[1], 100000);
      }
      // sm__inst_executed_pipe_tensor.sum
      //EXPECT_EQ(measurement.values[2], 0);
    }
  }
  return true;
}
#endif // HAS_CUPTI_RANGE_PROFILER

int main(int argc, char* argv[]) {

  CUdevice cuDevice;

  int deviceCount, deviceNum;
  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

  printf("Usage: %s [device_num]\n", argv[0]);

  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    LOG(ERROR) << "There is no device supporting CUDA.";
    return -2;
  }

  if (argc > 1)
    deviceNum = atoi(argv[1]);
  else
    deviceNum = 0;
  LOG(INFO) << "CUDA Device Number: " << deviceNum;

  DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMajor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      cuDevice));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMinor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      cuDevice));

  LOG(INFO) << "Compute Cabapbility = "
            << fmt::format("{},{}",computeCapabilityMajor, computeCapabilityMinor);

  if (computeCapabilityMajor < 7) {
    LOG(ERROR) << "CUPTI Profiler is not supported  with compute capability < 7.0";
    return -2;
  }

  CuptiRBProfilerSession::staticInit();

  // metrics to profile
  std::vector<std::string> metricNames = {
    "smsp__warps_launched.avg",
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__inst_executed_pipe_tensor.sum",
  };

  CUcontext cuContext;
  DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

  VectorAddSubtract();

#if HAS_CUPTI_RANGE_PROFILER
  CuptiRBProfilerSession::staticInit();

  if (!runTestWithUserRange(deviceNum, metricNames, cuContext, false)) {
    LOG(ERROR) << "Failed to profiler test benchmark in user range";
  } else if (!runTestWithAutoRange(deviceNum, metricNames, cuContext, false)) {
    LOG(ERROR) << "Failed to profiler test benchmark in auto range";
  } else if (!runTestWithUserRange(deviceNum, metricNames, cuContext, true)) {
    LOG(ERROR) << "Failed to profiler test benchmark in user range async";
  } else if (!runTestWithAutoRange(deviceNum, metricNames, cuContext, true)) {
    LOG(ERROR) << "Failed to profiler test benchmark in auto range async";
  }

  CuptiRBProfilerSession::deInitCupti();
#else
  LOG(WARNING) << "CuptiRBProfilerSession is not supported.";
#endif // HAS_CUPTI_RANGE_PROFILER
  DRIVER_API_CALL(cuCtxDestroy(cuContext));


  return 0;
}
