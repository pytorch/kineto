/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AsyncActivityProfilerHandler.h"

#include <chrono>

#include "ActivityProfilerController.h"
#include "Config.h"
#include "GenericActivityProfiler.h"
#include "Logger.h"
#include "ThreadUtil.h"
#include "libkineto.h"
#include "output_membuf.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

AsyncActivityProfilerHandler::AsyncActivityProfilerHandler(
    GenericActivityProfiler& profiler,
    ConfigLoader& configLoader)
    : configLoader_(configLoader), profiler_(profiler) {}

AsyncActivityProfilerHandler::~AsyncActivityProfilerHandler() {
  for (auto profilerThread : profilerThreads_) {
    if (profilerThread) {
      // signaling termination of the profiler loop
      stopRunloop_ = true;
      profilerThread->join();
      delete profilerThread;
      profilerThread = nullptr;
    }
  }
}

bool AsyncActivityProfilerHandler::canAcceptConfig() {
  return !profiler_.isActive();
}

void AsyncActivityProfilerHandler::acceptConfig(const Config& config) {
  VLOG(1) << "acceptConfig";
  if (config.activityProfilerEnabled()) {
    scheduleTrace(config);
  }
}

void AsyncActivityProfilerHandler::scheduleTrace(const Config& config) {
  VLOG(1) << "scheduleTrace";
  if (profiler_.isActive()) {
    LOG(WARNING) << "Ignored request - profiler busy";
    return;
  }

  int64_t currentIter = iterationCount_;
  std::unique_ptr<Config> configToSchedule;

  if (config.hasProfileStartIteration() && currentIter < 0) {
    // Special case: daemon config with activitiesDuration set
    if (config.activitiesDuration().count() > 0) {
      LOG(INFO) << "Config with duration-based profiling, "
                << "ignoring iteration count requirement";
      // Continue with modified config - clone and set profileStartIteration to
      // -1
      configToSchedule = config.clone();
      configToSchedule->setProfileStartIteration(-1);
    } else {
      LOG(WARNING) << "Ignored profile iteration count based request as "
                   << "application is not updating iteration count";
      return;
    }
  } else {
    configToSchedule = config.clone();
  }

  // Common scheduling logic
  bool newConfigScheduled = false;
  if (!asyncRequestConfig_) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    if (!asyncRequestConfig_) {
      asyncRequestConfig_ = std::move(configToSchedule);
      newConfigScheduled = true;
    }
  }
  if (!newConfigScheduled) {
    LOG(WARNING) << "Ignored request - another profile request is pending.";
    return;
  }

  // start a profilerLoop() thread to handle request

  if (config.memoryProfilerEnabled()) {
    auto thread_type = ThreadType::MEMORY_SNAPSHOT;
    if (!profilerThreads_[thread_type]) {
      profilerThreads_[thread_type] = new std::thread(
          &AsyncActivityProfilerHandler::memoryProfilerLoop, this);
    }
  } else {
    auto thread_type = ThreadType::KINETO;
    if (!profilerThreads_[thread_type]) {
      profilerThreads_[thread_type] =
          new std::thread(&AsyncActivityProfilerHandler::profilerLoop, this);
    }
  }
}

void AsyncActivityProfilerHandler::step() {
  // Do not remove this copy to currentIter. Otherwise count is not
  // guaranteed.
  int64_t currentIter = ++iterationCount_;
  VLOG(0) << "Step called , iteration  = " << currentIter;

  // Perform Double-checked locking to reduce overhead of taking lock.
  if (asyncRequestConfig_ && !profiler_.isActive()) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    auto now = system_clock::now();
    if (asyncRequestConfig_ && !profiler_.isActive() &&
        shouldActivateIterationConfig(currentIter)) {
      activateConfig(now);
    }
  }
  if (profiler_.isActive() && !profiler_.isCollectingMemorySnapshot()) {
    auto now = system_clock::now();
    auto next_wakeup_time = now + Config::kControllerIntervalMsecs;
    profiler_.performRunLoopStep(now, next_wakeup_time, currentIter);
  }
}

bool AsyncActivityProfilerHandler::shouldActivateTimestampConfig(
    const std::chrono::time_point<std::chrono::system_clock>& now) {
  if (asyncRequestConfig_->hasProfileStartIteration()) {
    return false;
  }
  if (asyncRequestConfig_->memoryProfilerEnabled()) {
    return false;
  }
  // Note on now + Config::kControllerIntervalMsecs:
  // Profiler interval does not align perfectly up to startTime - warmup.
  // Waiting until the next tick won't allow sufficient time for the
  // profiler to warm up. So check if we are very close to the warmup time
  // and trigger warmup.
  if (now + Config::kControllerIntervalMsecs >=
      (asyncRequestConfig_->requestTimestamp() -
       asyncRequestConfig_->activitiesWarmupDuration())) {
    LOG(INFO)
        << "Received on-demand activity trace request by "
        << " profile timestamp = "
        << asyncRequestConfig_->requestTimestamp().time_since_epoch().count();
    return true;
  }
  return false;
}

bool AsyncActivityProfilerHandler::shouldActivateIterationConfig(
    int64_t currentIter) {
  if (!asyncRequestConfig_->hasProfileStartIteration()) {
    return false;
  }
  if (asyncRequestConfig_->memoryProfilerEnabled()) {
    return false;
  }
  auto rootIter = asyncRequestConfig_->startIterationIncludingWarmup();
  // Keep waiting, it is not time to start yet.
  if (currentIter < rootIter) {
    return false;
  }

  LOG(INFO) << "Received on-demand activity trace request by "
               " profile start iteration = "
            << asyncRequestConfig_->profileStartIteration()
            << ", current iteration = " << currentIter;
  // Re-calculate the start iter if requested iteration is in the past.
  if (currentIter > rootIter) {
    auto newProfileStart =
        currentIter + asyncRequestConfig_->activitiesWarmupIterations();
    // Use Start Iteration Round Up if it is present.
    if (asyncRequestConfig_->profileStartIterationRoundUp() > 0) {
      // round up to nearest multiple
      auto divisor = asyncRequestConfig_->profileStartIterationRoundUp();
      auto rem = newProfileStart % divisor;
      newProfileStart += ((rem == 0) ? 0 : divisor - rem);
      LOG(INFO) << "Rounding up profiler start iteration to : "
                << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
      if (currentIter != asyncRequestConfig_->startIterationIncludingWarmup()) {
        // Ex. Current 9, start 8, warmup 5, roundup 100. Resolves new start
        // to 100, with warmup starting at 95. So don't start now.
        return false;
      }
    } else {
      LOG(INFO) << "Start iteration updated to : " << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
    }
  }
  return true;
}

void AsyncActivityProfilerHandler::profilerLoop() {
  setThreadName("Kineto Activity Profiler");
  VLOG(0) << "Entering activity profiler loop";

  auto now = system_clock::now();
  auto next_wakeup_time = now + Config::kControllerIntervalMsecs;

  while (!stopRunloop_) {
    now = system_clock::now();

    while (now < next_wakeup_time) {
      /* sleep override */
      std::this_thread::sleep_for(next_wakeup_time - now);
      now = system_clock::now();
    }

    // Perform Double-checked locking to reduce overhead of taking lock.
    if (asyncRequestConfig_ && !profiler_.isActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !profiler_.isActive() &&
          shouldActivateTimestampConfig(now)) {
        activateConfig(now);
      }
    }

    while (next_wakeup_time < now) {
      next_wakeup_time += Config::kControllerIntervalMsecs;
    }

    if (profiler_.isActive() && !profiler_.isCollectingMemorySnapshot()) {
      next_wakeup_time = profiler_.performRunLoopStep(now, next_wakeup_time);
      VLOG(1) << "Profiler loop: "
              << duration_cast<milliseconds>(system_clock::now() - now).count()
              << "ms";
    }
  }

  VLOG(0) << "Exited activity profiling loop";
}

void AsyncActivityProfilerHandler::memoryProfilerLoop() {
  while (!stopRunloop_) {
    // Perform Double-checked locking to reduce overhead of taking lock.
    if (asyncRequestConfig_ && !profiler_.isActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !profiler_.isActive() &&
          asyncRequestConfig_->memoryProfilerEnabled()) {
        logger_ = ActivityProfilerController::makeLogger(*asyncRequestConfig_);
        auto path = asyncRequestConfig_->activitiesLogFile();
        auto profile_time = asyncRequestConfig_->profileMemoryDuration();
        auto config = asyncRequestConfig_->clone();
        asyncRequestConfig_ = nullptr;
        profiler_.performMemoryLoop(path, profile_time, logger_.get(), *config);
      }
    }
  }
}

// This function should only be called when holding the configLock_.
void AsyncActivityProfilerHandler::activateConfig(
    std::chrono::time_point<std::chrono::system_clock> now) {
  logger_ = ActivityProfilerController::makeLogger(*asyncRequestConfig_);
  profiler_.setLogger(logger_.get());
  LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND();
  profiler_.configure(*asyncRequestConfig_, now);
  asyncRequestConfig_ = nullptr;
}

} // namespace KINETO_NAMESPACE
