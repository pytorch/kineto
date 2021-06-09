/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <set>
#include <vector>

#include "test/MockActivitySubProfiler.h"

namespace libkineto {

const std::set<ActivityType> supported_activities {ActivityType::CPU_OP};
const std::string profile_name{"MockProfiler"};

void MockProfilerSession::processTrace(ActivityLogger& logger) {
  for (const auto& activity: activities()) {
    activity.log(logger);
  }
}

const std::string& MockActivityProfiler::name() const {
  return profile_name;
}

const std::set<ActivityType>& MockActivityProfiler::availableActivities() const {
  return supported_activities;
}

MockActivityProfiler::MockActivityProfiler(
    std::vector<GenericTraceActivity>& activities) :
  test_activities_(activities) {};

std::unique_ptr<IActivityProfilerSession> MockActivityProfiler::configure(
      const std::set<ActivityType>& /*activity_types*/,
      const std::string& /*config*/) {
  auto session = std::make_unique<MockProfilerSession>();
	session->set_test_activities(std::move(test_activities_));
  return session;
};

std::unique_ptr<IActivityProfilerSession> MockActivityProfiler::configure(
      int64_t /*ts_ms*/,
      int64_t /*duration_ms*/,
      const std::set<ActivityType>& activity_types,
      const std::string& config) {
  return configure(activity_types, config);
};

} // namespace libkineto

