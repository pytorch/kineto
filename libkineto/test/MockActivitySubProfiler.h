/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "include/IActivityProfiler.h"

namespace libkineto {

class MockProfilerSession: public IActivityProfilerSession {

  public:
    explicit MockProfilerSession() {}

    void start() override {
      start_count++;
      status_ = TraceStatus::RECORDING;
    }

    void stop() override {
      stop_count++;
      status_ = TraceStatus::PROCESSING;
    }

    std::vector<GenericTraceActivity>& activities() override {
      return test_activities_;
    }

    std::vector<std::string> errors() override {
      return {};
    }

    void processTrace(ActivityLogger& logger) override;

    void set_test_activities(std::vector<GenericTraceActivity>&& acs) {
      test_activities_ = std::move(acs);
    }

    int start_count = 0;
    int stop_count = 0;
  private:
    std::vector<GenericTraceActivity> test_activities_;
};


class MockActivityProfiler: public IActivityProfiler {

 public:
  explicit MockActivityProfiler(std::vector<GenericTraceActivity>& activities);

  const std::string& name() const override;

  const std::set<ActivityType>& availableActivities() const override;

  std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const std::string& config = "") override;

  std::unique_ptr<IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const std::string& config = "") override;

 private:
  std::vector<GenericTraceActivity> test_activities_;
};

} // namespace libkineto
