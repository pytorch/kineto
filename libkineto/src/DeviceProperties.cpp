/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DeviceProperties.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <vector>

#ifdef HAS_CUPTI
#include <cuda_occupancy.h>
#include <cuda_runtime.h>
#elif defined(HAS_ROCTRACER)
#include <hip/hip_runtime.h>
#elif defined(HAS_XPUPTI)
#include "plugin/xpupti/XpuptiActivityProfiler.h"
#endif

#include "Logger.h"

namespace KINETO_NAMESPACE {

#ifdef HAS_CUPTI
#define gpuDeviceProp cudaDeviceProp
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#elif defined(HAS_ROCTRACER)
#define gpuDeviceProp hipDeviceProp_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#endif

#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
static std::vector<gpuDeviceProp> createDeviceProps() {
  std::vector<gpuDeviceProp> props;
  int device_count;
  gpuError_t error_id = gpuGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != gpuSuccess) {
    LOG(ERROR) << "gpuGetDeviceCount failed with code " << error_id;
    return {};
  }
  VLOG(0) << "Device count is " << device_count;
  for (int i = 0; i < device_count; ++i) {
    gpuDeviceProp prop;
    error_id = gpuGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != gpuSuccess) {
      LOG(ERROR) << "gpuGetDeviceProperties failed with " << error_id;
      return {};
    }
    props.push_back(prop);
    LOGGER_OBSERVER_ADD_DEVICE(i);
  }
  return props;
}

static const std::vector<gpuDeviceProp>& deviceProps() {
  static const std::vector<gpuDeviceProp> props = createDeviceProps();
  return props;
}

static std::string createDevicePropertiesJson(
    size_t id,
    const gpuDeviceProp& props) {
  std::string gpuSpecific;
#ifdef HAS_CUPTI
  gpuSpecific = fmt::format(
      R"JSON(
    , "regsPerMultiprocessor": {}, "sharedMemPerBlockOptin": {}, "sharedMemPerMultiprocessor": {})JSON",
      props.regsPerMultiprocessor,
      props.sharedMemPerBlockOptin,
      props.sharedMemPerMultiprocessor);
#elif defined(HAS_ROCTRACER)
  gpuSpecific = fmt::format(
      R"JSON(
    , "maxSharedMemoryPerMultiProcessor": {})JSON",
      props.maxSharedMemoryPerMultiProcessor);
#endif

  return fmt::format(
      R"JSON(
    {{
      "id": {}, "name": "{}", "totalGlobalMem": {},
      "computeMajor": {}, "computeMinor": {},
      "maxThreadsPerBlock": {}, "maxThreadsPerMultiprocessor": {},
      "regsPerBlock": {}, "warpSize": {},
      "sharedMemPerBlock": {}, "numSms": {}{}
    }})JSON",
      id,
      props.name,
      props.totalGlobalMem,
      props.major,
      props.minor,
      props.maxThreadsPerBlock,
      props.maxThreadsPerMultiProcessor,
      props.regsPerBlock,
      props.warpSize,
      props.sharedMemPerBlock,
      props.multiProcessorCount,
      gpuSpecific);
}

static std::string createDevicePropertiesJson() {
  std::vector<std::string> jsonProps;
  const auto& props = deviceProps();
  jsonProps.reserve(props.size());
  for (size_t i = 0; i < props.size(); i++) {
    jsonProps.push_back(createDevicePropertiesJson(i, props[i]));
  }
  return fmt::format("{}", fmt::join(jsonProps, ","));
}
#endif // HAS_CUPTI || HAS_ROCTRACER

const std::string& devicePropertiesJson() {
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  static std::string devicePropsJson = createDevicePropertiesJson();
#elif defined(HAS_XPUPTI)
  static std::string devicePropsJson = getXpuDeviceProperties();
#else
  static std::string devicePropsJson;
#endif
  return devicePropsJson;
}

int smCount([[maybe_unused]] uint32_t deviceId) {
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  const std::vector<gpuDeviceProp>& props = deviceProps();
  return deviceId >= props.size() ? 0 : props[deviceId].multiProcessorCount;
#else
  return 0;
#endif
}

#ifdef HAS_CUPTI
float blocksPerSm(const CUpti_ActivityKernelType& kernel) {
  int sm_count = smCount(kernel.deviceId);
  if (sm_count == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return (kernel.gridX * kernel.gridY * kernel.gridZ) /
      static_cast<float>(sm_count);
}

float warpsPerSm(const CUpti_ActivityKernelType& kernel) {
  constexpr int threads_per_warp = 32;
  return blocksPerSm(kernel) * (kernel.blockX * kernel.blockY * kernel.blockZ) /
      threads_per_warp;
}

OccupancyMetrics computeOccupancyMetrics(
    const CUpti_ActivityKernelType& kernel) {
  OccupancyMetrics metrics;
  const std::vector<cudaDeviceProp>& props = deviceProps();
  if (kernel.deviceId >= props.size()) {
    LOG(ERROR) << "Invalid deviceId " << kernel.deviceId
               << " exceeds available devices (" << props.size()
               << "), skipping occupancy calculation";
    return metrics;
  }

  float blocksPerSm = -1.0;
  int sm_count = smCount(kernel.deviceId);
  if (sm_count != 0) {
    blocksPerSm = (kernel.gridX * kernel.gridY * kernel.gridZ) /
        static_cast<float>(sm_count);
  }

  cudaOccFuncAttributes occFuncAttr;
  occFuncAttr.maxThreadsPerBlock = INT_MAX;
  occFuncAttr.numRegs = kernel.registersPerThread;
  occFuncAttr.sharedSizeBytes = kernel.staticSharedMemory;
  occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
  occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
  occFuncAttr.maxDynamicSharedSizeBytes = 0;
  const cudaOccDeviceState occDeviceState = {};
  int blockSize = kernel.blockX * kernel.blockY * kernel.blockZ;
  size_t dynamicSmemSize = kernel.dynamicSharedMemory;
  cudaOccDeviceProp prop(props[kernel.deviceId]);
  cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
      &metrics.result,
      &prop,
      &occFuncAttr,
      &occDeviceState,
      blockSize,
      dynamicSmemSize);
  if (status == CUDA_OCC_SUCCESS) {
    float effectiveBlocksPerSm = std::min<float>(
        metrics.result.activeBlocksPerMultiprocessor, blocksPerSm);
    metrics.occupancy = effectiveBlocksPerSm * blockSize /
        static_cast<float>(props[kernel.deviceId].maxThreadsPerMultiProcessor);
  } else {
    LOG_EVERY_N(ERROR, 1000)
        << "Failed to calculate occupancy, status = " << status;
  }
  return metrics;
}
#endif // HAS_CUPTI

} // namespace KINETO_NAMESPACE
