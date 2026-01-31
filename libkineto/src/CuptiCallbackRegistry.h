/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cupti.h>

namespace KINETO_NAMESPACE {

// Domain of the CUPTI callback
enum class CallbackDomain : uint8_t {
  RUNTIME, // CUDA Runtime API (cudaXxx functions)
  DRIVER, // CUDA Driver API (cuXxx functions)
};

// CuptiCallbackRegistry: Central registry for CUPTI callback metadata
//
// This class provides a single source of truth for all CUPTI callback
// properties, replacing scattered hardcoded checks throughout the codebase.
//
// The registry is initialized lazily on first access via instance().
// All callbacks are registered in the constructor with their properties.
//
// Usage:
//   auto& registry = CuptiCallbackRegistry::instance();
//   if (registry.requiresFlowCorrelation(CallbackDomain::RUNTIME, cbid)) {
//     // Create flow correlation
//   }
//
class CuptiCallbackRegistry {
 public:
  // Get the singleton instance (lazy initialization on first call)
  static CuptiCallbackRegistry& instance();

  // Disable copy/move
  CuptiCallbackRegistry(const CuptiCallbackRegistry&) = delete;
  CuptiCallbackRegistry& operator=(const CuptiCallbackRegistry&) = delete;
  CuptiCallbackRegistry(CuptiCallbackRegistry&&) = delete;
  CuptiCallbackRegistry& operator=(CuptiCallbackRegistry&&) = delete;

  // Check if a callback requires flow correlation (CPU->GPU arrows in trace)
  bool requiresFlowCorrelation(CallbackDomain domain, uint32_t cbid) const;

  // Check if a callback is blocklisted (should be filtered from traces)
  bool isBlocklisted(CallbackDomain domain, uint32_t cbid) const;

 private:
  CuptiCallbackRegistry();
  ~CuptiCallbackRegistry() = default;

  // Properties stored per callback
  struct CallbackProps {
    bool requiresFlowCorrelation;
    bool isBlocklisted;
  };

  // Range of callbacks (for memory operations)
  struct CallbackRange {
    uint32_t startCbid;
    uint32_t endCbid; // inclusive
    bool requiresFlowCorrelation;
  };

  // Register a callback
  void registerCallback(
      CallbackDomain domain,
      uint32_t cbid,
      bool requiresFlowCorrelation,
      bool isBlocklisted);

  // Register a range of callbacks
  void registerCallbackRange(
      CallbackDomain domain,
      uint32_t startCbid,
      uint32_t endCbid,
      bool requiresFlowCorrelation);

  // Storage per domain
  std::unordered_map<uint32_t, CallbackProps> runtimeCallbacks_;
  std::unordered_map<uint32_t, CallbackProps> driverCallbacks_;

  // Ranges for callbacks that use range-based matching
  std::vector<std::pair<CallbackDomain, CallbackRange>> callbackRanges_;

  // Helper to get the appropriate map
  const std::unordered_map<uint32_t, CallbackProps>* getMapForDomain(
      CallbackDomain domain) const;
};

} // namespace KINETO_NAMESPACE
