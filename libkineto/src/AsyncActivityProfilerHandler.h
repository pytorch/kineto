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

  void acceptConfig(const Config& config);
  void scheduleTrace(const Config& config);
  void step();

  [[nodiscard]] bool isAsyncActive() const {
    return currentRunloopState_ != RunloopState::WaitForRequest;
  }

  [[nodiscard]] bool isCollectingMemorySnapshot() const {
    return currentRunloopState_ == RunloopState::CollectMemorySnapshot;
  }

  void cancel();

  void configure(const Config& config, std::chrono::time_point<std::chrono::system_clock> now);

  // Invoke at a regular interval to perform profiling activities.
  // When not active, an interval of 1-5 seconds is probably fine,
  // depending on required warm-up time and delayed start time.
  // When active, it's a good idea to invoke more frequently to stay below
  // memory usage limit (ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB) during warmup.
  std::chrono::time_point<std::chrono::system_clock> performRunLoopStep(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime,
      int64_t currentIter = -1);

  void ensureCollectTraceDone();

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

  enum class RunloopState {
    WaitForRequest,
    Warmup,
    CollectTrace,
    ProcessTrace,
    CollectMemorySnapshot,
  };

  void performMemoryLoop(const std::string& path, uint32_t profile_time, ActivityLogger* logger, Config& config);

  void collectTrace(bool collection_done, const std::chrono::time_point<std::chrono::system_clock>& now);

  bool getCollectTraceState();

  std::atomic<RunloopState> currentRunloopState_{RunloopState::WaitForRequest};
  std::unique_ptr<std::thread> collectTraceThread_{nullptr};
  std::recursive_mutex collectTraceStateMutex_;
  bool isCollectingTrace_{false};
};
} // namespace KINETO_NAMESPACE
