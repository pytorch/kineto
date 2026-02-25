/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "ActivityTraceInterface.h"
#include "GenericActivityProfiler.h"

namespace KINETO_NAMESPACE {

class Config;

class SyncActivityProfilerHandler {
 public:
  explicit SyncActivityProfilerHandler(GenericActivityProfiler& profiler);

  SyncActivityProfilerHandler(const SyncActivityProfilerHandler&) = delete;
  SyncActivityProfilerHandler& operator=(const SyncActivityProfilerHandler&) =
      delete;

  ~SyncActivityProfilerHandler() = default;

  void prepareTrace(const Config& config);
  void toggleCollectionDynamic(const bool enable);
  void startTrace();
  std::unique_ptr<ActivityTraceInterface> stopTrace();

 private:
  GenericActivityProfiler& profiler_;
};
} // namespace KINETO_NAMESPACE
