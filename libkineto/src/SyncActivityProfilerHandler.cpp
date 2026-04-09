/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SyncActivityProfilerHandler.h"

#include <chrono>

#include "ActivityProfilerController.h"
#include "ActivityTrace.h"
#include "Config.h"
#include "GenericActivityProfiler.h"
#include "Logger.h"
#include "libkineto.h"
#include "output_membuf.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

SyncActivityProfilerHandler::SyncActivityProfilerHandler(
    GenericActivityProfiler& profiler)
    : profiler_(profiler) {}

void SyncActivityProfilerHandler::prepareTrace(const Config& config) {
  auto now = std::chrono::system_clock::now();
  if (!profiler_.canStart(config, now)) {
    return;
  }
  profiler_.configure(config, now);
  active_ = true;
}

void SyncActivityProfilerHandler::startTrace() {
  UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
  profiler_.startTrace(std::chrono::system_clock::now());
}

std::unique_ptr<ActivityTraceInterface> SyncActivityProfilerHandler::
    stopTrace() {
  profiler_.stopTrace(std::chrono::system_clock::now());
  UST_LOGGER_MARK_COMPLETED(kCollectionStage);
  auto logger = std::make_unique<MemoryTraceLogger>(profiler_.config());
  profiler_.processTrace(*logger);
  // Will follow up with another patch for logging URLs when ActivityTrace
  // is moved.
  UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);

  // Logger Metadata contains a map of LOGs collected in Kineto
  //   logger_level -> List of log lines
  // This will be added into the trace as metadata.
  std::unordered_map<std::string, std::vector<std::string>> loggerMD =
      profiler_.getLoggerMetadata();
  logger->setLoggerMetadata(std::move(loggerMD));

  profiler_.reset();
  active_ = false;
  return std::make_unique<ActivityTrace>(
      std::move(logger), ActivityProfilerController::loggerFactory());
}

void SyncActivityProfilerHandler::cancel() {
  if (!active_) {
    return;
  }
  if (libkineto::api().client()) {
    libkineto::api().client()->stop();
  }
  profiler_.stopTraceAndReset(std::chrono::system_clock::now());
  active_ = false;
}

void SyncActivityProfilerHandler::toggleCollectionDynamic(const bool enable) {
  profiler_.toggleCollectionDynamic(enable);
}

} // namespace KINETO_NAMESPACE
