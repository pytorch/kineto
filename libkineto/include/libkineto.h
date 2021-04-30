/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mediator for initialization and profiler control

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>

#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "ClientInterface.h"
#include "GenericTraceActivity.h"
#include "TraceSpan.h"

#include "ThreadUtil.h"

extern "C" {
  void suppressLibkinetoLogMessages();
  bool libkineto_init(bool cpuOnly, bool logOnError);
}

namespace libkineto {

class Config;

struct CpuTraceBuffer {
  TraceSpan span;
  int gpuOpCount;
  std::vector<GenericTraceActivity> activities;
};

class LibkinetoApi {
 public:

  // Called by client that supports tracing API.
  // libkineto can still function without this.
  void registerClient(ClientInterface* client);

  // Called by libkineto on init
  void registerProfiler(std::unique_ptr<ActivityProfilerInterface> profiler) {
    activityProfiler_ = std::move(profiler);
    initClientIfRegistered();
  }

  ActivityProfilerInterface& activityProfiler() {
    return *activityProfiler_;
  }

  ClientInterface* client() {
    return client_;
  }

  void initProfilerIfRegistered() {
    if (activityProfiler_ && !activityProfiler_->isInitialized()) {
      activityProfiler_->init();
    }
  }

  bool isProfilerInitialized() const {
    return activityProfiler_ && activityProfiler_->isInitialized();
  }

  bool isProfilerRegistered() const {
    return activityProfiler_ != nullptr;
  }

  void setNetSizeThreshold(int gpu_ops) {
    netSizeThreshold_ = gpu_ops;
  }

  // Inclide traces with at least this many ops
  // FIXME: Rename and move elsewhere
  int netSizeThreshold() {
    return netSizeThreshold_;
  }

  void suppressLogMessages() {
    suppressLibkinetoLogMessages();
  }

 private:

  // Client is initialized once both it and libkineto has registered
  void initClientIfRegistered();

  std::unique_ptr<ActivityProfilerInterface> activityProfiler_{};
  ClientInterface* client_{};
  int32_t clientRegisterThread_{0};

  bool isLoaded_{false};
  std::atomic_int netSizeThreshold_{};
};

// Singleton
LibkinetoApi& api();

} // namespace libkineto
