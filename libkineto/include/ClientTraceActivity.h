/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <thread>

#include "TraceActivity.h"

namespace libkineto {

struct ClientTraceActivity : TraceActivity {
  ClientTraceActivity() = default;
  ClientTraceActivity(ClientTraceActivity&&) = default;
  ClientTraceActivity& operator=(ClientTraceActivity&&) = default;
  ~ClientTraceActivity() override {}

  int64_t deviceId() const override {
    return cachedPid();
  }

  int64_t resourceId() const override {
    return sysThreadId;
  }

  int64_t timestamp() const override {
    return startTime;
  }

  int64_t duration() const override {
    return endTime - startTime;
  }

  int64_t correlationId() const override {
    return correlation;
  }

  ActivityType type() const override {
    return ActivityType::CPU_OP;
  }

  const std::string name() const override {
    return opType;
  }

  const TraceActivity* linkedActivity() const override {
    return nullptr;
  }

  void log(ActivityLogger& logger) const override {
    // Unimplemented by default
  }

  int64_t startTime{0};
  int64_t endTime{0};
  int64_t correlation{0};
  int device{-1};
  // TODO: Add OS abstraction
  pthread_t pthreadId{};
  int32_t sysThreadId{0};
  std::string opType;
  std::string inputDims;
  std::string inputTypes;
  std::string arguments;
  std::string outputDims;
  std::string outputTypes;
  std::string inputNames;
  std::string outputNames;
  std::string callStack;
};

} // namespace libkineto
