/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerController.h"

#include <chrono>
#include <thread>

#include "ActivityTrace.h"
#include "CuptiActivityInterface.h"
#include "ThreadUtil.h"
#include "output_json.h"
#include "output_membuf.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr milliseconds kProfilerIntervalMsecs(1000);

ActivityProfilerController::ActivityProfilerController(bool cpuOnly) {
  profiler_ = std::make_unique<ActivityProfiler>(CuptiActivityInterface::singleton(), cpuOnly);
}

ActivityProfilerController::~ActivityProfilerController() {
  if (profilerThread_) {
    // signaling termination of the profiler loop
    stopRunloop_ = true;
    profilerThread_->join();
    delete profilerThread_;
    profilerThread_ = nullptr;
  }
  VLOG(0) << "Stopped activity profiler";
}

static ActivityLoggerFactory& loggerFactory() {
  static ActivityLoggerFactory factory{nullptr};
  return factory;
}

void ActivityProfilerController::setLoggerFactory(
    const ActivityLoggerFactory& factory) {
  loggerFactory() = factory;
}

static std::unique_ptr<ActivityLogger> makeLogger(const Config& config) {
  if (config.activitiesLogToMemory()) {
    return std::make_unique<MemoryTraceLogger>(config);
  }
  if (loggerFactory()) {
    return loggerFactory()(config);
  }
  return std::make_unique<ChromeTraceLogger>(
      config.activitiesLogFile(),
      CuptiActivityInterface::singleton().smCount());
}

void ActivityProfilerController::profilerLoop() {
  setThreadName("Kineto Activity Profiler");
  VLOG(0) << "Entering activity profiler loop";

  auto now = system_clock::now();
  auto next_wakeup_time = now + kProfilerIntervalMsecs;

  while (!stopRunloop_) {
    now = system_clock::now();

    while (now < next_wakeup_time) {
      /* sleep override */
      std::this_thread::sleep_for(next_wakeup_time - now);
      now = system_clock::now();
    }
    if (!profiler_->isActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_) {
        LOG(INFO) << "Received on-demand activity trace request";
        logger_ = makeLogger(*asyncRequestConfig_);
        profiler_->setLogger(logger_.get());
        profiler_->configure(*asyncRequestConfig_, now);
        asyncRequestConfig_ = nullptr;
      }
    }

    while (next_wakeup_time < now) {
      next_wakeup_time += kProfilerIntervalMsecs;
    }

    if (profiler_->isActive()) {
      next_wakeup_time = profiler_->performRunLoopStep(now, next_wakeup_time);
      VLOG(1) << "Profiler loop: "
          << duration_cast<milliseconds>(system_clock::now() - now).count()
          << "ms";
    }
  }

  VLOG(0) << "Exited activity profiling loop";
}

void ActivityProfilerController::scheduleTrace(const Config& config) {
  std::lock_guard<std::mutex> lock(asyncConfigLock_);
  asyncRequestConfig_ = config.clone();
  // start a profilerLoop() thread to handle request
  if (!profilerThread_) {
    profilerThread_ =
        new std::thread(&ActivityProfilerController::profilerLoop, this);
  }
}

void ActivityProfilerController::prepareTrace(const Config& config) {
  // Requests from ActivityProfilerApi have higher priority than
  // requests from other sources (signal, daemon).
  // Cancel any ongoing request and refuse new ones.
  auto now = system_clock::now();
  if (profiler_->isActive()) {
    LOG(WARNING) << "Cancelling current trace request in order to start "
                 << "higher priority synchronous request";
    if (libkineto::api().client()) {
      libkineto::api().client()->stop();
    }
    profiler_->stopTrace(now);
    profiler_->reset();
  }

  profiler_->configure(config, now);
}

std::unique_ptr<ActivityTraceInterface> ActivityProfilerController::stopTrace() {
  if (libkineto::api().client()) {
    libkineto::api().client()->stop();
  }
  profiler_->stopTrace(std::chrono::system_clock::now());
  auto logger = std::make_unique<MemoryTraceLogger>(profiler_->config());
  profiler_->processTrace(*logger);
  profiler_->reset();
  return std::make_unique<ActivityTrace>(std::move(logger), CuptiActivityInterface::singleton());
}

void ActivityProfilerController::addMetadata(
    const std::string& key, const std::string& value) {
  profiler_->addMetadata(key, value);
}

} // namespace KINETO_NAMESPACE
