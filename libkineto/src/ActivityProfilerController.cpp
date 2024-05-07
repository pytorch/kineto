/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerController.h"

#include <chrono>
#include <functional>
#include <thread>

#include "ActivityLoggerFactory.h"
#include "ActivityTrace.h"

#include "CuptiActivityApi.h"
#ifdef HAS_ROCTRACER
#include "RoctracerActivityApi.h"
#endif

#include "ThreadUtil.h"
#include "output_json.h"
#include "output_membuf.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

#if !USE_GOOGLE_LOG
static std::shared_ptr<LoggerCollector>& loggerCollectorFactory() {
  static std::shared_ptr<LoggerCollector> factory = nullptr;
  return factory;
}

void ActivityProfilerController::setLoggerCollectorFactory(
    std::function<std::shared_ptr<LoggerCollector>()> factory) {
  loggerCollectorFactory() = factory();
}
#endif // !USE_GOOGLE_LOG

ActivityProfilerController::ActivityProfilerController(
    ConfigLoader& configLoader, bool cpuOnly)
    : configLoader_(configLoader) {
  // Initialize ChromeTraceBaseTime first of all.
  ChromeTraceBaseTime::singleton().init();

#if !USE_GOOGLE_LOG
  // Initialize LoggerCollector before ActivityProfiler to log
  // CUPTI and CUDA driver versions.
  if (loggerCollectorFactory()) {
    // Keep a reference to the logger collector factory to handle safe
    // static de-initialization.
    loggerCollectorFactory_ = loggerCollectorFactory();
    Logger::addLoggerObserver(loggerCollectorFactory_.get());
  }
#endif // !USE_GOOGLE_LOG

#ifdef HAS_ROCTRACER
  profiler_ = std::make_unique<CuptiActivityProfiler>(
      RoctracerActivityApi::singleton(), cpuOnly);
#else
  profiler_ = std::make_unique<CuptiActivityProfiler>(
      CuptiActivityApi::singleton(), cpuOnly);
#endif
  configLoader_.addHandler(ConfigLoader::ConfigKind::ActivityProfiler, this);
}

ActivityProfilerController::~ActivityProfilerController() {
  configLoader_.removeHandler(
      ConfigLoader::ConfigKind::ActivityProfiler, this);
  if (profilerThread_) {
    // signaling termination of the profiler loop
    stopRunloop_ = true;
    profilerThread_->join();
    delete profilerThread_;
    profilerThread_ = nullptr;
  }

#if !USE_GOOGLE_LOG
  if (loggerCollectorFactory()) {
    Logger::removeLoggerObserver(loggerCollectorFactory_.get());
  }
#endif // !USE_GOOGLE_LOG
}

static ActivityLoggerFactory initLoggerFactory() {
  ActivityLoggerFactory factory;
  factory.addProtocol("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
  });
  return factory;
}

static ActivityLoggerFactory& loggerFactory() {
  static ActivityLoggerFactory factory = initLoggerFactory();
  return factory;
}

void ActivityProfilerController::addLoggerFactory(
    const std::string& protocol, ActivityLoggerFactory::FactoryFunc factory) {
  loggerFactory().addProtocol(protocol, factory);
}

static std::unique_ptr<ActivityLogger> makeLogger(const Config& config) {
  if (config.activitiesLogToMemory()) {
    return std::make_unique<MemoryTraceLogger>(config);
  }
  return loggerFactory().makeLogger(config.activitiesLogUrl());
}

static std::unique_ptr<InvariantViolationsLogger>& invariantViolationsLoggerFactory() {
  static std::unique_ptr<InvariantViolationsLogger> factory = nullptr;
  return factory;
}

void ActivityProfilerController::setInvariantViolationsLoggerFactory(
    const std::function<std::unique_ptr<InvariantViolationsLogger>()>& factory) {
  invariantViolationsLoggerFactory() = factory();
}

bool ActivityProfilerController::canAcceptConfig() {
  return !profiler_->isActive();
}

void ActivityProfilerController::acceptConfig(const Config& config) {
  VLOG(1) << "acceptConfig";
  if (config.activityProfilerEnabled()) {
    scheduleTrace(config);
  }
}

bool ActivityProfilerController::shouldActivateTimestampConfig(
    const std::chrono::time_point<std::chrono::system_clock>& now) {
  if (asyncRequestConfig_->hasProfileStartIteration()) {
    return false;
  }
  // Note on now + Config::kControllerIntervalMsecs:
  // Profiler interval does not align perfectly up to startTime - warmup.
  // Waiting until the next tick won't allow sufficient time for the profiler to warm up.
  // So check if we are very close to the warmup time and trigger warmup.
  if (now + Config::kControllerIntervalMsecs
      >= (asyncRequestConfig_->requestTimestamp() - asyncRequestConfig_->activitiesWarmupDuration())) {
    LOG(INFO) << "Received on-demand activity trace request by "
              << " profile timestamp = "
              << asyncRequestConfig_->requestTimestamp().time_since_epoch().count();
    return true;
  }
  return false;
}

bool ActivityProfilerController::shouldActivateIterationConfig(
    int64_t currentIter) {
  if (!asyncRequestConfig_->hasProfileStartIteration()) {
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
    auto newProfileStart = currentIter +
        asyncRequestConfig_->activitiesWarmupIterations();
    // Use Start Iteration Round Up if it is present.
    if (asyncRequestConfig_->profileStartIterationRoundUp() > 0) {
      // round up to nearest multiple
      auto divisor = asyncRequestConfig_->profileStartIterationRoundUp();
      auto rem = newProfileStart % divisor;
      newProfileStart += ((rem == 0) ? 0 : divisor - rem);
      LOG(INFO) << "Rounding up profiler start iteration to : " << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
      if (currentIter != asyncRequestConfig_->startIterationIncludingWarmup()) {
        // Ex. Current 9, start 8, warmup 5, roundup 100. Resolves new start to 100,
        // with warmup starting at 95. So don't start now.
        return false;
      }
    } else {
      LOG(INFO) << "Start iteration updated to : " << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
    }
  }
  return true;
}

void ActivityProfilerController::profilerLoop() {
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
    if (asyncRequestConfig_ && !profiler_->isActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !profiler_->isActive() &&
          shouldActivateTimestampConfig(now)) {
        activateConfig(now);
      }
    }

    while (next_wakeup_time < now) {
      next_wakeup_time += Config::kControllerIntervalMsecs;
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

void ActivityProfilerController::step() {
  // Do not remove this copy to currentIter. Otherwise count is not guaranteed.
  int64_t currentIter = ++iterationCount_;
  VLOG(0) << "Step called , iteration  = " << currentIter;

  // Perform Double-checked locking to reduce overhead of taking lock.
  if (asyncRequestConfig_ && !profiler_->isActive()) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    auto now = system_clock::now();
    if (asyncRequestConfig_ && !profiler_->isActive() &&
        shouldActivateIterationConfig(currentIter)) {
      activateConfig(now);
    }
  }

  if (profiler_->isActive()) {
    auto now = system_clock::now();
    auto next_wakeup_time = now + Config::kControllerIntervalMsecs;
    profiler_->performRunLoopStep(now, next_wakeup_time, currentIter);
  }
}

// This function should only be called when holding the configLock_.
void ActivityProfilerController::activateConfig(
    std::chrono::time_point<std::chrono::system_clock> now) {
  logger_ = makeLogger(*asyncRequestConfig_);
  profiler_->setLogger(logger_.get());
  LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND();
  profiler_->configure(*asyncRequestConfig_, now);
  asyncRequestConfig_ = nullptr;
}

void ActivityProfilerController::scheduleTrace(const Config& config) {
  VLOG(1) << "scheduleTrace";
  if (profiler_->isActive()) {
    LOG(WARNING) << "Ignored request - profiler busy";
    return;
  }
  int64_t currentIter = iterationCount_;
  if (config.hasProfileStartIteration() && currentIter < 0) {
    LOG(WARNING) << "Ignored profile iteration count based request as "
                 << "application is not updating iteration count";
    return;
  }

  bool newConfigScheduled = false;
  if (!asyncRequestConfig_) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    if (!asyncRequestConfig_) {
      asyncRequestConfig_ = config.clone();
      newConfigScheduled = true;
    }
  }
  if (!newConfigScheduled) {
    LOG(WARNING) << "Ignored request - another profile request is pending.";
    return;
  }

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

void ActivityProfilerController::startTrace() {
  UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
  profiler_->startTrace(std::chrono::system_clock::now());
}

std::unique_ptr<ActivityTraceInterface> ActivityProfilerController::stopTrace() {
  profiler_->stopTrace(std::chrono::system_clock::now());
  UST_LOGGER_MARK_COMPLETED(kCollectionStage);
  auto logger = std::make_unique<MemoryTraceLogger>(profiler_->config());
  profiler_->processTrace(*logger);
  // Will follow up with another patch for logging URLs when ActivityTrace is moved.
  UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);

  // Logger Metadata contains a map of LOGs collected in Kineto
  //   logger_level -> List of log lines
  // This will be added into the trace as metadata.
  std::unordered_map<std::string, std::vector<std::string>>
    loggerMD = profiler_->getLoggerMetadata();
  logger->setLoggerMetadata(std::move(loggerMD));

  profiler_->reset();
  return std::make_unique<ActivityTrace>(std::move(logger), loggerFactory());
}

void ActivityProfilerController::addMetadata(
    const std::string& key, const std::string& value) {
  profiler_->addMetadata(key, value);
}

void ActivityProfilerController::logInvariantViolation(
    const std::string& profile_id,
    const std::string& assertion,
    const std::string& error,
    const std::string& group_profile_id) {
  if (invariantViolationsLoggerFactory()) {
    invariantViolationsLoggerFactory()->logInvariantViolation(profile_id, assertion, error, group_profile_id);
  }
}

} // namespace KINETO_NAMESPACE
