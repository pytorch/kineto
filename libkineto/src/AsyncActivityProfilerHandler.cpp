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
    GenericActivityProfiler& profiler)
    : profiler_(profiler) {}

AsyncActivityProfilerHandler::~AsyncActivityProfilerHandler() {
  ensureCollectTraceDone();
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

void AsyncActivityProfilerHandler::acceptConfig(const Config& config) {
  VLOG(1) << "acceptConfig";
  if (config.activityProfilerEnabled()) {
    scheduleTrace(config);
  }
}

void AsyncActivityProfilerHandler::scheduleTrace(const Config& config) {
  VLOG(1) << "scheduleTrace";

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
  if (asyncRequestConfig_ && !isAsyncActive()) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    auto now = system_clock::now();
    if (asyncRequestConfig_ && !isAsyncActive() &&
        shouldActivateIterationConfig(currentIter)) {
      activateConfig(now);
    }
  }
  if (isAsyncActive() && !isCollectingMemorySnapshot()) {
    auto now = system_clock::now();
    auto next_wakeup_time = now + Config::kControllerIntervalMsecs;
    performRunLoopStep(now, next_wakeup_time, currentIter);
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
    if (asyncRequestConfig_ && !isAsyncActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !isAsyncActive() &&
          shouldActivateTimestampConfig(now)) {
        activateConfig(now);
      }
    }

    while (next_wakeup_time < now) {
      next_wakeup_time += Config::kControllerIntervalMsecs;
    }

    if (isAsyncActive() && !isCollectingMemorySnapshot()) {
      next_wakeup_time = performRunLoopStep(now, next_wakeup_time);
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
    if (asyncRequestConfig_ && !isAsyncActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !isAsyncActive() &&
          asyncRequestConfig_->memoryProfilerEnabled()) {
        logger_ = ActivityProfilerController::makeLogger(*asyncRequestConfig_);
        auto path = asyncRequestConfig_->activitiesLogFile();
        auto profile_time = asyncRequestConfig_->profileMemoryDuration();
        auto config = asyncRequestConfig_->clone();
        asyncRequestConfig_ = nullptr;
        performMemoryLoop(path, profile_time, logger_.get(), *config);
      }
    }
  }
}

void AsyncActivityProfilerHandler::configure(
    const Config& config,
    std::chrono::time_point<std::chrono::system_clock> now) {
  if (!profiler_.canStart(config, now)) {
    return;
  }
  logger_ = ActivityProfilerController::makeLogger(config);
  profiler_.setLogger(logger_.get());
  LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND();
  profiler_.configure(config, now);
  currentRunloopState_ = RunloopState::Warmup;
}

// This function should only be called when holding the configLock_.
void AsyncActivityProfilerHandler::activateConfig(
    std::chrono::time_point<std::chrono::system_clock> now) {
  configure(*asyncRequestConfig_, now);
  asyncRequestConfig_ = nullptr;
}

time_point<system_clock> AsyncActivityProfilerHandler::performRunLoopStep(
    const time_point<system_clock>& now,
    const time_point<system_clock>& nextWakeupTime,
    int64_t currentIter) {
  auto new_wakeup_time = nextWakeupTime;

  VLOG_IF(1, currentIter >= 0)
      << "Run loop on application step(), iteration = " << currentIter;

  switch (currentRunloopState_) {
    case RunloopState::CollectMemorySnapshot:
      LOG(WARNING)
          << "Entered CollectMemorySnapshot in Kineto Loop Step, skipping loop";
      break;
    case RunloopState::WaitForRequest:
      VLOG(1) << "State: WaitForRequest";
      break;
    case RunloopState::Cancelling:
      // cancel() is tearing down the profiler on another thread.
      // Do nothing — we must not drive the profiler concurrently.
      VLOG(1) << "State: Cancelling";
      break;

    case RunloopState::Warmup: {
      VLOG(1) << "State: Warmup";
      profiler_.flushWarmupBuffers(currentIter, nextWakeupTime);

      if (profiler_.isGpuCollectionStopped()) {
        profiler_.stopTraceAndReset(now);
        LOG(ERROR)
            << "State: Warmup stopped by GPU profiler. (Buffer size configured is "
            << profiler_.activitiesMaxGpuBufferSizeMB() << "MB)";
        UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
        VLOG(0) << "Warmup -> WaitForRequest";
        currentRunloopState_ = RunloopState::WaitForRequest;
        break;
      }

      if (profiler_.isWarmupDone(now, currentIter)) {
        UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
        if (!profiler_.isProfilingByIteration() &&
            (now > profiler_.profileStartTime() + milliseconds(10))) {
          LOG(INFO) << "Tracing started "
                    << duration_cast<milliseconds>(
                           now - profiler_.profileStartTime())
                           .count()
                    << "ms late!";
        } else {
          LOG(INFO) << "Tracing started";
        }
        profiler_.startTrace(now);
        currentRunloopState_ = RunloopState::CollectTrace;
        if (libkineto::api().client()) {
          libkineto::api().client()->start();
        }
        if (nextWakeupTime > profiler_.profileEndTime()) {
          new_wakeup_time = profiler_.profileEndTime();
        }
      } else if (nextWakeupTime > profiler_.profileStartTime()) {
        new_wakeup_time = profiler_.profileStartTime();
      }
      break;
    }

    case RunloopState::CollectTrace: {
      VLOG(1) << "State: CollectTrace";
      bool collection_done = profiler_.isCollectionDone(now, currentIter);

      if (collection_done || profiler_.isGpuCollectionStopped()) {
        LOG(INFO) << "Tracing complete.";
        VLOG_IF(1, currentIter >= 0)
            << "This state change was invoked by application's step() call";
        // currentIter >= 0 means this is called from the step() api of
        // the profiler in pytorch main thread, it should be executed in
        // another thread in case pytorch main thread is blocked
        if (currentIter >= 0) {
          // if collectTraceThread_ is already running, there's no need to
          // execute collectTrace twice.
          // Do not call collectTrace when profilerThread_ is collecting
          // Trace. Otherwise, libkineto::api().client()->stop will be called
          // twice, which leads to an unrecoverable ::c10:Error at
          // disableProfiler
          if (!collectTraceThread_ && !getCollectTraceState()) {
            std::lock_guard<std::recursive_mutex> guard(
                collectTraceStateMutex_);
            collectTraceThread_ = std::make_unique<std::thread>(
                &AsyncActivityProfilerHandler::collectTrace,
                this,
                collection_done,
                now);
          }
          break;
        }
        // this is executed in profilerThread_
        {
          std::lock_guard<std::recursive_mutex> guard(collectTraceStateMutex_);
          isCollectingTrace_ = true;
        }
        collectTrace(collection_done, now);
        {
          std::lock_guard<std::recursive_mutex> guard(collectTraceStateMutex_);
          isCollectingTrace_ = false;
        }
      } else if (profiler_.isProfilingByIteration()) {
        // nothing to do here
      } else if (
          now < profiler_.profileEndTime() &&
          profiler_.profileEndTime() < nextWakeupTime) {
        new_wakeup_time = profiler_.profileEndTime();
      }
      break;
    }

    case RunloopState::ProcessTrace: {
      VLOG(1) << "State: ProcessTrace";
      // skip this state transition if it called from the step() api
      // of the profiler.
      // else it could lead to a race between the profiler thread and an
      // application thread calling step()
      if (currentIter >= 0) {
        return new_wakeup_time;
      }

      // Before processing, we should wait for collectTrace thread to be done.
      ensureCollectTraceDone();

      // FIXME: Probably want to allow interruption here
      // for quickly handling trace request via synchronous API
      profiler_.processTraceAndReset(*logger_);
      UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);
      currentRunloopState_ = RunloopState::WaitForRequest;
      VLOG(0) << "ProcessTrace -> WaitForRequest";
      break;
    }
  }

  return new_wakeup_time;
}

void AsyncActivityProfilerHandler::collectTrace(
    bool collection_done,
    const std::chrono::time_point<std::chrono::system_clock>& now) {
  profiler_.collectTrace(collection_done, now);
  currentRunloopState_ = RunloopState::ProcessTrace;
}

void AsyncActivityProfilerHandler::performMemoryLoop(
    const std::string& path,
    uint32_t profile_time,
    ActivityLogger* logger,
    Config& config) {
  currentRunloopState_ = RunloopState::CollectMemorySnapshot;
  if (libkineto::api().client()) {
    libkineto::api().client()->start_memory_profile();
    LOG(INFO) << "Running memory profiling for " << profile_time << " ms";
    std::this_thread::sleep_for(std::chrono::milliseconds(profile_time));
    LOG(INFO) << "Exporting memory profiling results to " << path;
    libkineto::api().client()->export_memory_profile(path);
    libkineto::api().client()->stop_memory_profile();
    LOG(INFO) << "Finalizing trace";
    logger->finalizeMemoryTrace(path, config);
  }
  currentRunloopState_ = RunloopState::WaitForRequest;
}

bool AsyncActivityProfilerHandler::getCollectTraceState() {
  std::lock_guard<std::recursive_mutex> guard(collectTraceStateMutex_);
  return isCollectingTrace_;
}

void AsyncActivityProfilerHandler::ensureCollectTraceDone() {
  if (collectTraceThread_ && collectTraceThread_->joinable()) {
    collectTraceThread_->join();
    collectTraceThread_.reset(nullptr);
  }
}

void AsyncActivityProfilerHandler::cancel() {
  {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    asyncRequestConfig_ = nullptr;
  }
  if (!isAsyncActive()) {
    return;
  }
  currentRunloopState_ = RunloopState::Cancelling;
  ensureCollectTraceDone();
  if (libkineto::api().client()) {
    libkineto::api().client()->stop();
  }
  profiler_.stopTraceAndReset(std::chrono::system_clock::now());
  currentRunloopState_ = RunloopState::WaitForRequest;
}

} // namespace KINETO_NAMESPACE
