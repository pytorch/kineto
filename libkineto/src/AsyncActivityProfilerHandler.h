/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include "ActivityLoggerFactory.h"
#include "GenericActivityProfiler.h"

namespace KINETO_NAMESPACE {

enum ThreadType {
  KINETO = 0,
  MEMORY_SNAPSHOT,
  THREAD_MAX_COUNT // Number of enum entries (used for array sizing)
};

class Config;

class AsyncActivityProfilerHandler {
 public:
  explicit AsyncActivityProfilerHandler(GenericActivityProfiler& profiler, std::atomic_bool& syncTraceActive);

  AsyncActivityProfilerHandler(const AsyncActivityProfilerHandler&) = delete;
  AsyncActivityProfilerHandler& operator=(const AsyncActivityProfilerHandler&) = delete;

  ~AsyncActivityProfilerHandler();

  bool canAcceptConfig();
  void acceptConfig(const Config& config);
  void scheduleTrace(const Config& config);
  void step();

 private:
  bool shouldActivateIterationConfig(int64_t currentIter);
  bool shouldActivateTimestampConfig(const std::chrono::time_point<std::chrono::system_clock>& now);
  void profilerLoop();
  void memoryProfilerLoop();
  void activateConfig(std::chrono::time_point<std::chrono::system_clock> now);

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;
  std::thread* profilerThreads_[ThreadType::THREAD_MAX_COUNT] = {nullptr};
  std::atomic_bool stopRunloop_{false};
  std::atomic<std::int64_t> iterationCount_{-1};

  GenericActivityProfiler& profiler_;
  std::atomic_bool& syncTraceActive_;
  std::unique_ptr<ActivityLogger> logger_;
};
} // namespace KINETO_NAMESPACE
