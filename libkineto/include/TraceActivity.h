/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include "ActivityType.h"

namespace libkineto {

class ActivityLogger;
struct TraceSpan;

// Generic activity interface is borrowed from tensorboard protobuf format.
struct TraceActivity {
  virtual ~TraceActivity() {}
  // Device is a physical or logical entity, e.g. CPU, GPU or process
  virtual int64_t deviceId() const = 0;
  // A resource is something on the device, e.g. s/w or h/w thread,
  // functional units etc.
  virtual int64_t resourceId() const = 0;
  // Start timestamp in mucrosecond
  virtual int64_t timestamp() const = 0;
  // Duration in microseconds
  virtual int64_t duration() const = 0;
  // Used to link up async activities
  virtual int64_t correlationId() const = 0;
  virtual ActivityType type() const = 0;
  virtual const std::string name() const = 0;
  // Optional linked activity
  virtual const TraceActivity* linkedActivity() const = 0;
  // Optional containing trace object
  virtual const TraceSpan* traceSpan() const = 0;
  // Log activity
  virtual void log(ActivityLogger& logger) const = 0;

  static int64_t nsToUs(int64_t ns) {
    // It's important that this conversion is the same everywhere.
    // No rounding!
    return ns / 1000;
  }
};

} // namespace libkineto
