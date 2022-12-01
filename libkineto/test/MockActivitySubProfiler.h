/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <deque>

#include "include/IActivityProfiler.h"
#include "output_base.h"

namespace libkineto {

class MockProfilerSession: public IActivityProfilerSession {

 public:
  virtual ~MockProfilerSession() {}

  std::mutex& mutex() override {
    return mutex_;
  }

  TraceStatus status() override {
    return status_;
  }

  void status(TraceStatus status) override {
    status_ = status;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void log(ActivityLogger& logger) override;

  void setTestActivities(std::vector<const ITraceActivity*>* acs) {
    testActivities_ = acs;
  }

  const std::vector<const ITraceActivity*>* activities() {
    return testActivities_;
  }

private:
  TraceStatus status_;
  std::mutex mutex_;
  std::vector<const ITraceActivity*>* testActivities_;
};


class MockActivityProfiler: public IActivityProfiler {

 public:
  explicit MockActivityProfiler(std::deque<GenericTraceActivity>& activities);

  virtual ~MockActivityProfiler() {};

  const std::string name() const override;

  const std::set<ActivityType>& supportedActivityTypes() const override;

  void init(ICompositeProfiler* parent) = 0;
  bool isInitialized() const = 0;
  bool isActive() const = 0;

  std::shared_ptr<IActivityProfilerSession> configure(
      const Config& options,
      ICompositeProfilerSession* parentSession) override;

  void start(IActivityProfilerSession& session) override {
    start_count++;
    session_->status(TraceStatus::RECORDING);
  }

  void stop(IActivityProfilerSession& session) override {
    stop_count++;
    session_->status(TraceStatus::PROCESSING);
  }

  int start_count = 0;
  int stop_count = 0;

 private:
  std::shared_ptr<MockProfilerSession> session_;
};

} // namespace libkineto
