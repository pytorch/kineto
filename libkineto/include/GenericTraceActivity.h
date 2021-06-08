/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <string>
#include <thread>
#include <vector>

#include "ThreadUtil.h"
#include "TraceActivity.h"
#include "TraceSpan.h"

namespace libkineto {

// @lint-ignore-every CLANGTIDY cppcoreguidelines-non-private-member-variables-in-classes
// @lint-ignore-every CLANGTIDY cppcoreguidelines-pro-type-member-init
class GenericTraceActivity : public TraceActivity {

 public:
  GenericTraceActivity() = delete;

  GenericTraceActivity(
      const TraceSpan& trace, ActivityType type, const std::string& name)
      : activityType(type), activityName(name), traceSpan_(&trace) {
  }

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
    return id;
  }

  ActivityType type() const override {
    return activityType;
  }

  const std::string name() const override {
    return activityName;
  }

  const TraceActivity* linkedActivity() const override {
    return nullptr;
  }

  const TraceSpan* traceSpan() const override {
    return traceSpan_;
  }

  void log(ActivityLogger& logger) const override;

  //Encode client side metadata as a key/value string.
  void addMetadata(const std::string& key, const std::string& value) {
    metadata_.push_back(fmt::format("\"{}\": {}", key, value));
  }

  const std::string getMetadata() const {
    return fmt::format("{}", fmt::join(metadata_, ", "));
  }

  virtual ~GenericTraceActivity() {};

  int64_t startTime{0};
  int64_t endTime{0};
  int32_t id{0};
  int32_t device{0};
  int32_t resource{0};
  ActivityType activityType;
  std::string activityName;

 private:
  const TraceSpan* traceSpan_;
  std::vector<std::string> metadata_;
};

} // namespace libkineto
