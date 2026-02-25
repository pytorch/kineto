/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO DEVICE_AGNOSTIC: The build system should be strict enough that we don't
//                       need this guard in the header file.
#ifdef HAS_ROCTRACER

#include <roctracer.h>
#include "GenericActivityProfiler.h"
#include "RoctracerActivityApi.h"
#include "RoctracerLogger.h"

namespace KINETO_NAMESPACE {

class RocmActivityProfiler : public GenericActivityProfiler {
 public:
  RocmActivityProfiler(RoctracerActivityApi& roctracer, bool cpuOnly);
  RocmActivityProfiler(const RocmActivityProfiler&) = delete;
  RocmActivityProfiler& operator=(const RocmActivityProfiler&) = delete;
  ~RocmActivityProfiler() override = default;

 protected:
  void logGpuVersions() override;
  void setMaxGpuBufferSize(int size) override;
  void enableGpuTracing() override;
  void disableGpuTracing() override;
  void clearGpuActivities() override;
  bool isGpuCollectionStopped() const override;
  void processGpuActivities(ActivityLogger& logger) override;
  void synchronizeGpuDevice() override;
  void pushCorrelationIdImpl(uint64_t id, CorrelationFlowType type) override;
  void popCorrelationIdImpl(CorrelationFlowType type) override;
  void onResetTraceData() override;
  void onFinalizeTrace(const Config& config, ActivityLogger& logger) override;

 private:
  // Process generic RocTracer activity
  void handleRoctracerActivity(const roctracerBase* record, ActivityLogger* logger);
  void handleCorrelationActivity(uint64_t correlationId,
                                 uint64_t externalId,
                                 RoctracerLogger::CorrelationDomain externalKind);
  // Process specific GPU activity types
  template <class T>
  void handleRuntimeActivity(const T* activity, ActivityLogger* logger);
  void handleGpuActivity(const roctracerAsyncRow* record, ActivityLogger* logger);

  // Calls to ROCtracer is encapsulated behind this interface
  RoctracerActivityApi& roctracer_;
};

} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
