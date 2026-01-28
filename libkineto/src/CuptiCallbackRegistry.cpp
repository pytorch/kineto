/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiCallbackRegistry.h"

namespace KINETO_NAMESPACE {

CuptiCallbackRegistry& CuptiCallbackRegistry::instance() {
  static CuptiCallbackRegistry instance;
  return instance;
}

const std::unordered_map<uint32_t, CuptiCallbackRegistry::CallbackProps>*
CuptiCallbackRegistry::getMapForDomain(CallbackDomain domain) const {
  switch (domain) {
    case CallbackDomain::RUNTIME:
      return &runtimeCallbacks_;
    case CallbackDomain::DRIVER:
      return &driverCallbacks_;
    default:
      return nullptr;
  }
}

void CuptiCallbackRegistry::registerCallback(
    CallbackDomain domain,
    uint32_t cbid,
    bool requiresFlowCorrelation,
    bool isBlocklisted) {
  CallbackProps props{requiresFlowCorrelation, isBlocklisted};
  switch (domain) {
    case CallbackDomain::RUNTIME:
      runtimeCallbacks_[cbid] = props;
      break;
    case CallbackDomain::DRIVER:
      driverCallbacks_[cbid] = props;
      break;
    default:
      break;
  }
}

void CuptiCallbackRegistry::registerCallbackRange(
    CallbackDomain domain,
    uint32_t startCbid,
    uint32_t endCbid,
    bool requiresFlowCorrelation) {
  callbackRanges_.push_back(
      {domain, CallbackRange{startCbid, endCbid, requiresFlowCorrelation}});
}

CuptiCallbackRegistry::CuptiCallbackRegistry() {
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
      /*requiresFlowCorrelation=*/true);

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
      /*isBlocklisted=*/false);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11060
  registerCallback(
      /*domain=*/CallbackDomain::DRIVER,
      /*cbid=*/CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
      /*requiresFlowCorrelation=*/true,
      /*isBlocklisted=*/false);
#endif
}

bool CuptiCallbackRegistry::requiresFlowCorrelation(
    CallbackDomain domain,
    uint32_t cbid) const {
  // Check explicit callbacks first
  const auto* map = getMapForDomain(domain);
  if (map != nullptr) {
    auto it = map->find(cbid);
    if (it != map->end()) {
      return it->second.requiresFlowCorrelation;
    }
  }
  // Check ranges (for memory operations)
  for (const auto& [rangeDomain, range] : callbackRanges_) {
    if (rangeDomain == domain && cbid >= range.startCbid &&
        cbid <= range.endCbid) {
      return range.requiresFlowCorrelation;
    }
  }
  return false;
}

bool CuptiCallbackRegistry::isBlocklisted(CallbackDomain domain, uint32_t cbid)
    const {
  const auto* map = getMapForDomain(domain);
  if (map != nullptr) {
    auto it = map->find(cbid);
    if (it != map->end()) {
      return it->second.isBlocklisted;
    }
  }
  return false;
}

} // namespace KINETO_NAMESPACE
