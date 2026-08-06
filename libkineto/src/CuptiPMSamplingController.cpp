/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiPMSamplingController.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

#include "Logger.h"

/*
 * ======================== CUPTI PM SAMPLING CONTROLLER
 * ========================
 *
 * Controller for PM Sampling, which manages Kineto-side lifecycle. Hooks onto
 * the CuptiPMSamplingApi. A sampling session follows prepare() -> start() ->
 * stop().
 *
 * TODO LIST (in no particular order):
 *   1. Bound sample counts for large traces.
 *   2. Handle on-demand/async profiling, plus interaction with auto-trace/sync.
 *   3. Push hard-coded constants into user-provided vars.
 *   4. Add tests.
 *
 * ==============================================================================
 */

namespace KINETO_NAMESPACE {
namespace {

constexpr std::chrono::milliseconds kDecodePollInterval{10};

} // namespace

CuptiPMSamplingController::CuptiPMSamplingController(
    CuptiPMSamplingConfig config,
    CuptiPMSamplingApi& api)
    : config_(std::move(config)), api_(api) {}

CuptiPMSamplingController::~CuptiPMSamplingController() {
  stop();
}

bool CuptiPMSamplingController::prepare() {
  if (prepared_) {
    LOG(WARNING) << "CUPTI PM sampling is busy";
    return false;
  }

  try {
    {
      // Prevent the decode thread from appending to samples while this is being
      // cleared
      std::lock_guard<std::mutex> lock(samplesMutex_);
      samples_.clear();
    }
    if (!validateConfig()) {
      return false;
    }
    api_.configure(config_);
  } catch (...) {
    logCurrentException("Failed to prepare CUPTI PM sampling");
    api_.disable();
    return false;
  }
  prepared_ = true;
  return true;
}

bool CuptiPMSamplingController::start() {
  if (!prepared_) {
    LOG(WARNING) << "CUPTI PM sampling is not prepared";
    return false;
  }
  if (active_) {
    LOG(WARNING) << "CUPTI PM sampling is already running";
    return false;
  }

  stopRequested_.store(false, std::memory_order_relaxed);
  try {
    api_.start();
    active_ = true;
    // Decode needs to happen on an async thread as it needs to synchronously
    // fetch new samples from the api
    decodeThread_ = std::thread(&CuptiPMSamplingController::decodeLoop, this);
  } catch (...) {
    logCurrentException("Failed to start CUPTI PM sampling");
    teardown();
  }
  return active_;
}

void CuptiPMSamplingController::stop() {
  if (!prepared_) {
    return;
  }
  if (active_) {
    stopRequested_.store(true, std::memory_order_relaxed);
    waitCondition_.notify_one();
    decodeThread_.join();
  }
  teardown();
}

const std::vector<std::string>& CuptiPMSamplingController::metricNames() const {
  return config_.metricNames;
}

std::vector<CuptiPMSample> CuptiPMSamplingController::samples() const {
  std::lock_guard<std::mutex> lock(samplesMutex_);
  return samples_;
}

void CuptiPMSamplingController::decodeLoop() {
  std::vector<CuptiPMSample> decodedSamples;
  try {
    // stopRequested_ flag signals that the main thread is stopping, we
    // use the waitCondition_ to wake the decode thread to return immediately.
    while (!stopRequested_.load(std::memory_order_relaxed)) {
      if (decodeBatch(decodedSamples)) {
        std::unique_lock<std::mutex> lock(waitMutex_);
        waitCondition_.wait_for(lock, kDecodePollInterval, [this]() {
          return stopRequested_.load(std::memory_order_relaxed);
        });
      }
    }
  } catch (...) {
    logCurrentException("CUPTI PM sampling worker failed");
  }
}

bool CuptiPMSamplingController::decodeBatch(
    std::vector<CuptiPMSample>& decodedSamples) {
  decodedSamples.clear();
  const bool isBufferDrained = api_.decode(decodedSamples);
  std::lock_guard<std::mutex> lock(samplesMutex_);
  for (auto& sample : decodedSamples) {
    if (validateSample(sample)) {
      samples_.push_back(std::move(sample));
    }
  }
  return isBufferDrained;
}

void CuptiPMSamplingController::drain(
    std::vector<CuptiPMSample>& decodedSamples) {
  // Called at the end to fully drain the hardware buffer
  bool isBufferDrained;
  do {
    isBufferDrained = decodeBatch(decodedSamples);
  } while (!isBufferDrained);
}

void CuptiPMSamplingController::teardown() {
  if (active_) {
    std::vector<CuptiPMSample> decodedSamples;
    try {
      api_.stop();
      drain(decodedSamples);
    } catch (...) {
      logCurrentException("Failed to stop or drain CUPTI PM sampling");
    }
    active_ = false;
  }
  api_.disable();
  prepared_ = false;
}

bool CuptiPMSamplingController::validateConfig() const {
  if (config_.deviceId < 0) {
    LOG(WARNING) << "CUPTI PM sampling device ID must be nonnegative";
    return false;
  }
  if (config_.metricNames.empty() ||
      std::any_of(
          config_.metricNames.begin(),
          config_.metricNames.end(),
          [](const std::string& name) { return name.empty(); })) {
    LOG(WARNING) << "CUPTI PM sampling metrics must not be empty";
    return false;
  }
  if (config_.samplingInterval.count() <= 0) {
    LOG(WARNING) << "CUPTI PM sampling interval must be positive";
    return false;
  }
  return true;
}

bool CuptiPMSamplingController::validateSample(
    const CuptiPMSample& sample) const {
  if (sample.values.size() != config_.metricNames.size()) {
    LOG(WARNING) << "CUPTI PM sample has the wrong number of metric values";
    return false;
  }
  if (sample.rawEndTimestamp < sample.rawStartTimestamp) {
    LOG(WARNING) << "CUPTI PM sample has an invalid timestamp interval";
    return false;
  }
  if (std::any_of(sample.values.begin(), sample.values.end(), [](double value) {
        return !std::isfinite(value);
      })) {
    LOG(WARNING) << "CUPTI PM sample contains a non-finite metric value";
    return false;
  }
  return true;
}

void CuptiPMSamplingController::logCurrentException(const char* fallback) {
  const auto exception = std::current_exception();
  if (exception == nullptr) {
    LOG(WARNING) << fallback;
    return;
  }
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& error) {
    LOG(WARNING) << error.what();
  } catch (...) {
    LOG(WARNING) << fallback;
  }
}

} // namespace KINETO_NAMESPACE
