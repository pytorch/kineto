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
    GenericActivityProfiler& profiler,
    std::atomic_bool& syncTraceActive)
    : profiler_(profiler), syncTraceActive_(syncTraceActive) {}

void SyncActivityProfilerHandler::prepareTrace(const Config& config) {
  // Requests from ActivityProfilerApi have higher priority than
  // requests from other sources (signal, daemon).
  // Cancel any ongoing request and refuse new ones.
  auto now = system_clock::now();
  syncTraceActive_ = true;
  if (profiler_.isActive()) {
    LOG(WARNING) << "Cancelling current trace request in order to start "
                 << "higher priority synchronous request";
    if (libkineto::api().client()) {
      libkineto::api().client()->stop();
    }

    profiler_.stopTrace(now);
    profiler_.reset();
  }

  profiler_.configure(config, now);
}

void SyncActivityProfilerHandler::startTrace() {
  UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
  USDT_EMIT_START_TRACE();
  profiler_.startTrace(std::chrono::system_clock::now());
}

std::unique_ptr<ActivityTraceInterface> SyncActivityProfilerHandler::
    stopTrace() {
  profiler_.stopTrace(std::chrono::system_clock::now());
  USDT_EMIT_STOP_TRACE();
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
  syncTraceActive_ = false;
  return std::make_unique<ActivityTrace>(
      std::move(logger), ActivityProfilerController::loggerFactory());
}

void SyncActivityProfilerHandler::toggleCollectionDynamic(const bool enable) {
  profiler_.toggleCollectionDynamic(enable);
}

} // namespace KINETO_NAMESPACE
