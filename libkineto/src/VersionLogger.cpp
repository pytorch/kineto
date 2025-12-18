/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <VersionLogger.h>

#ifdef HAS_XPUPTI
#include "plugin/xpupti/XpuptiVersionLogger.h"
#endif

namespace KINETO_NAMESPACE {

void CudaVersionLogger::logAndRecordVersions() {
#ifdef HAS_CUPTI
  // check Nvidia versions
  uint32_t cuptiVersion = 0;
  int cudaRuntimeVersion = 0, cudaDriverVersion = 0;
  CUPTI_CALL(cuptiGetVersion(&cuptiVersion));
  CUDA_CALL(cudaRuntimeGetVersion(&cudaRuntimeVersion));
  CUDA_CALL(cudaDriverGetVersion(&cudaDriverVersion));
  LOG(INFO) << "CUDA versions. CUPTI: " << cuptiVersion
            << "; Runtime: " << cudaRuntimeVersion
            << "; Driver: " << cudaDriverVersion;

  LOGGER_OBSERVER_ADD_METADATA("cupti_version", std::to_string(cuptiVersion));
  LOGGER_OBSERVER_ADD_METADATA(
      "cuda_runtime_version", std::to_string(cudaRuntimeVersion));
  LOGGER_OBSERVER_ADD_METADATA(
      "cuda_driver_version", std::to_string(cudaDriverVersion));
  addVersionMetadata("cupti_version", std::to_string(cuptiVersion));
  addVersionMetadata(
      "cuda_runtime_version", std::to_string(cudaRuntimeVersion));
  addVersionMetadata("cuda_driver_version", std::to_string(cudaDriverVersion));
#endif
};

void HipVersionLogger::logAndRecordVersions() {
#ifdef HAS_ROCTRACER
  uint32_t majorVersion = roctracer_version_major();
  uint32_t minorVersion = roctracer_version_minor();
  std::string roctracerVersion =
      std::to_string(majorVersion) + "." + std::to_string(minorVersion);
  int hipRuntimeVersion = 0, hipDriverVersion = 0;
  CUDA_CALL(hipRuntimeGetVersion(&hipRuntimeVersion));
  CUDA_CALL(hipDriverGetVersion(&hipDriverVersion));
  LOG(INFO) << "HIP versions. Roctracer: " << roctracerVersion
            << "; Runtime: " << hipRuntimeVersion
            << "; Driver: " << hipDriverVersion;

  LOGGER_OBSERVER_ADD_METADATA("roctracer_version", roctracerVersion);
  LOGGER_OBSERVER_ADD_METADATA(
      "hip_runtime_version", std::to_string(hipRuntimeVersion));
  LOGGER_OBSERVER_ADD_METADATA(
      "hip_driver_version", std::to_string(hipDriverVersion));
  addVersionMetadata("roctracer_version", roctracerVersion);
  addVersionMetadata("hip_runtime_version", std::to_string(hipRuntimeVersion));
  addVersionMetadata("hip_driver_version", std::to_string(hipDriverVersion));
#endif
}

std::unique_ptr<DeviceVersionLogger> selectDeviceVersionLogger(
    std::recursive_mutex& mutex) {
#ifdef HAS_CUPTI
  return std::make_unique<CudaVersionLogger>(mutex);
#elif HAS_ROCTRACER
  return std::make_unique<HipVersionLogger>(mutex);
#elif HAS_XPUPTI
  return std::make_unique<XpuVersionLogger>(mutex);
#else
  return nullptr;
#endif
}

} // namespace KINETO_NAMESPACE
