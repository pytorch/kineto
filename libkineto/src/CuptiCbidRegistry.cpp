/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiCbidRegistry.h"

#include <stdexcept>

namespace KINETO_NAMESPACE {

CuptiCbidRegistry& CuptiCbidRegistry::instance() {
  static CuptiCbidRegistry instance;
  return instance;
}

std::unordered_map<uint32_t, CuptiCbidRegistry::CbidProperties>&
CuptiCbidRegistry::getMapForDomain(CallbackDomain domain) {
  switch (domain) {
    case CallbackDomain::RUNTIME:
      return runtimeCallbacks_;
    case CallbackDomain::DRIVER:
      return driverCallbacks_;
    default:
      throw std::invalid_argument("Unknown CallbackDomain");
  }
}

void CuptiCbidRegistry::registerCallback(
    CallbackDomain domain,
    uint32_t cbid,
    bool requiresFlowCorrelation,
    bool isBlocklisted) {
  getMapForDomain(domain)[cbid] =
      CbidProperties{requiresFlowCorrelation, isBlocklisted, {}};
}

void CuptiCbidRegistry::registerCallback(
    CallbackDomain domain,
    uint32_t cbid,
    bool requiresFlowCorrelation,
    bool isBlocklisted,
    const std::string& name) {
  getMapForDomain(domain)[cbid] =
      CbidProperties{requiresFlowCorrelation, isBlocklisted, name};
}

void CuptiCbidRegistry::registerCallbackRange(
    CallbackDomain domain,
    uint32_t startCbid,
    uint32_t endCbid,
    bool requiresFlowCorrelation,
    bool isBlocklisted) {
  cbidRanges_.push_back(
      {domain,
       CbidRange{startCbid, endCbid, requiresFlowCorrelation, isBlocklisted}});
}

CuptiCbidRegistry::CuptiCbidRegistry() {
  // =========================================================================
  // RUNTIME API - Kernel Launches
  // =========================================================================
  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);
#endif

  // =========================================================================
  // RUNTIME API - CUDA Graph Operations
  // =========================================================================
  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  // =========================================================================
  // RUNTIME API - Synchronization Operations
  // =========================================================================
  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  // =========================================================================
  // RUNTIME API - Memory Operations (range-based)
  // =========================================================================
  registerCallbackRange(
      /*domain=*/CallbackDomain::RUNTIME,
      /*startCbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
      /*endCbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);

  // =========================================================================
  // RUNTIME API - Blocklisted (noisy) Callbacks
  // =========================================================================
  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  registerCallback(
      /*domain=*/CallbackDomain::RUNTIME,
      /*cbid=*/CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/true);

  // =========================================================================
  // DRIVER API - Kernel Launches
  // =========================================================================
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false,
      /*name=*/"cuLaunchKernel");
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11060
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false,
      /*name=*/"cuLaunchKernelEx");
#endif
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemCreate,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemCreate");
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemMap,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemMap");
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemUnmap,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemUnmap");
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemRelease,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemRelease");
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemExportToShareableHandle");
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle,
      /*requiresFlowCorrelation=*/false,
      /*isBlocklisted=*/false,
      /*name=*/"cuMemImportFromShareableHandle");
}

bool CuptiCbidRegistry::requiresFlowCorrelation(
    CallbackDomain domain,
    uint32_t cbid) {
  // Check explicit callback IDs first
  const auto& map = getMapForDomain(domain);
  if (auto it = map.find(cbid); it != map.end()) {
    return it->second.requiresFlowCorrelation;
  }
  // Check ranges (for memory operations)
  // TODO: reevaluate the loop once we iterate of many ranges
  for (const auto& [rangeDomain, range] : cbidRanges_) {
    if (rangeDomain == domain && cbid >= range.startCbid &&
        cbid <= range.endCbid) {
      return range.requiresFlowCorrelation;
    }
  }
  return false;
}

bool CuptiCbidRegistry::isBlocklisted(CallbackDomain domain, uint32_t cbid) {
  // Check explicit callbacks first
  const auto& map = getMapForDomain(domain);
  if (auto it = map.find(cbid); it != map.end()) {
    return it->second.isBlocklisted;
  }
  // Check ranges
  // TODO: reevaluate the loop once we iterate of many ranges
  for (const auto& [rangeDomain, range] : cbidRanges_) {
    if (rangeDomain == domain && cbid >= range.startCbid &&
        cbid <= range.endCbid) {
      return range.isBlocklisted;
    }
  }
  return false;
}

bool CuptiCbidRegistry::isRegistered(CallbackDomain domain, uint32_t cbid) {
  const auto& map = getMapForDomain(domain);
  if (map.find(cbid) != map.end()) {
    return true;
  }
  for (const auto& [rangeDomain, range] : cbidRanges_) {
    if (rangeDomain == domain && cbid >= range.startCbid &&
        cbid <= range.endCbid) {
      return true;
    }
  }
  return false;
}

const std::string& CuptiCbidRegistry::getName(
    CallbackDomain domain,
    uint32_t cbid) {
  static const std::string unknown = "unknown";
  const auto& map = getMapForDomain(domain);
  if (auto it = map.find(cbid); it != map.end() && !it->second.name.empty()) {
    return it->second.name;
  }
  return unknown;
}

} // namespace KINETO_NAMESPACE
