/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerController.h"

#include <functional>
#include <utility>

#include "ActivityLoggerFactory.h"
// TODO DEVICE_AGNOSTIC: Move the device decision out of C++ files to be
//                       determined entirely by the build process. For the
//                       controller, we'll need some registration mechanism.
#ifdef HAS_CUPTI
#include "CuptiActivityApi.h"
#include "CuptiActivityProfiler.h"
#endif

#ifdef HAS_ROCTRACER
#include "RocmActivityProfiler.h"
#include "RoctracerActivityApi.h"
#endif

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
    const std::function<std::shared_ptr<LoggerCollector>()>& factory) {
  loggerCollectorFactory() = factory();
}

std::shared_ptr<LoggerCollector>
ActivityProfilerController::getLoggerCollector() {
  return loggerCollectorFactory();
}
#endif // !USE_GOOGLE_LOG

ActivityProfilerController::ActivityProfilerController(
    ConfigLoader& configLoader,
    bool cpuOnly)
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
  profiler_ = std::make_unique<RocmActivityProfiler>(
      RoctracerActivityApi::singleton(), cpuOnly);
#elif defined(HAS_CUPTI)
  profiler_ = std::make_unique<CuptiActivityProfiler>(
      CuptiActivityApi::singleton(), cpuOnly);
#else
  // CPU-only profiling without any GPU backends is handled
  // directly by the GenericActivityProfiler.
  profiler_ = std::make_unique<GenericActivityProfiler>(cpuOnly);
#endif
  configLoader_.addHandler(ConfigLoader::ConfigKind::ActivityProfiler, this);

  syncHandler_ = std::make_unique<SyncActivityProfilerHandler>(*profiler_);
  asyncHandler_ =
      std::make_unique<AsyncActivityProfilerHandler>(*profiler_, configLoader_);
}

ActivityProfilerController::~ActivityProfilerController() {
  configLoader_.removeHandler(ConfigLoader::ConfigKind::ActivityProfiler, this);
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

ActivityLoggerFactory& ActivityProfilerController::loggerFactory() {
  static ActivityLoggerFactory factory = initLoggerFactory();
  return factory;
}

void ActivityProfilerController::addLoggerFactory(
    const std::string& protocol,
    ActivityLoggerFactory::FactoryFunc factory) {
  loggerFactory().addProtocol(protocol, std::move(factory));
}

std::unique_ptr<ActivityLogger> ActivityProfilerController::makeLogger(
    const Config& config) {
  if (config.activitiesLogToMemory()) {
    return std::make_unique<MemoryTraceLogger>(config);
  }
  return loggerFactory().makeLogger(config.activitiesLogUrl());
}

static std::unique_ptr<InvariantViolationsLogger>&
invariantViolationsLoggerFactory() {
  static std::unique_ptr<InvariantViolationsLogger> factory = nullptr;
  return factory;
}

void ActivityProfilerController::setInvariantViolationsLoggerFactory(
    const std::function<std::unique_ptr<InvariantViolationsLogger>()>&
        factory) {
  invariantViolationsLoggerFactory() = factory();
}

bool ActivityProfilerController::isActive() {
  return profiler_->isActive();
}

void ActivityProfilerController::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  profiler_->transferCpuTrace(std::move(cpuTrace));
}

void ActivityProfilerController::recordThreadInfo() {
  profiler_->recordThreadInfo();
}

void ActivityProfilerController::addChildActivityProfiler(
    std::unique_ptr<IActivityProfiler> profiler) {
  profiler_->addChildActivityProfiler(std::move(profiler));
}

void ActivityProfilerController::pushCorrelationId(uint64_t id) {
  profiler_->pushCorrelationId(id);
}

void ActivityProfilerController::popCorrelationId() {
  profiler_->popCorrelationId();
}

void ActivityProfilerController::pushUserCorrelationId(uint64_t id) {
  profiler_->pushUserCorrelationId(id);
}

void ActivityProfilerController::popUserCorrelationId() {
  profiler_->popUserCorrelationId();
}

void ActivityProfilerController::addMetadata(
    const std::string& key,
    const std::string& value) {
  profiler_->addMetadata(key, value);
}

void ActivityProfilerController::logInvariantViolation(
    const std::string& profile_id,
    const std::string& assertion,
    const std::string& error,
    const std::string& group_profile_id) {
  if (invariantViolationsLoggerFactory()) {
    invariantViolationsLoggerFactory()->logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
}

// Async-only functions
bool ActivityProfilerController::canAcceptConfig() {
  return asyncHandler_->canAcceptConfig();
}
void ActivityProfilerController::acceptConfig(const Config& config) {
  asyncHandler_->acceptConfig(config);
}
void ActivityProfilerController::scheduleTrace(const Config& config) {
  asyncHandler_->scheduleTrace(config);
}
void ActivityProfilerController::step() {
  asyncHandler_->step();
}

// Sync-only functions
void ActivityProfilerController::prepareTrace(const Config& config) {
  syncHandler_->prepareTrace(config);
}
void ActivityProfilerController::toggleCollectionDynamic(const bool enable) {
  syncHandler_->toggleCollectionDynamic(enable);
}
void ActivityProfilerController::startTrace() {
  syncHandler_->startTrace();
}
std::unique_ptr<ActivityTraceInterface>
ActivityProfilerController::stopTrace() {
  return syncHandler_->stopTrace();
}

} // namespace KINETO_NAMESPACE
