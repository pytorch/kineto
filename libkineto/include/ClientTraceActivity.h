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
    return threadId;
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

  int64_t startTime;
  int64_t endTime;
  int64_t correlation;
  int device;
  pthread_t threadId;
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
