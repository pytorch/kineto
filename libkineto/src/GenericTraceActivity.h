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

namespace KINETO_NAMESPACE {

// A generic trace activity that can be freely modified
struct GenericTraceActivity : libkineto::TraceActivity {
  int64_t deviceId() const override {
    return device;
  }

  int64_t resourceId() const override {
    return resource;
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

  libkineto::ActivityType type() const override {
    return activityType;
  }

  const std::string name() const override {
    return activityName;
  }

  const libkineto::TraceActivity* linkedActivity() const override {
    return linked;
  }

  void log(libkineto::ActivityLogger& logger) const override;

  int64_t device;
  pthread_t resource;

  int64_t startTime;
  int64_t endTime;
  int64_t correlation;

  libkineto::ActivityType activityType;
  std::string activityName;

  libkineto::TraceActivity* linked;
};

} // namespace KINETO_NAMESPACE
