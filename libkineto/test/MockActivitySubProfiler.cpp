/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <set>
#include <vector>

#include "output_base.h"
#include "test/MockActivitySubProfiler.h"

namespace libkineto {

const std::set<ActivityType> supported_activities {ActivityType::CPU_OP};
const std::string profile_name{"MockProfiler"};

void MockProfilerSession::log(ActivityLogger& logger) {
  for (const auto& activity : *testActivities_) {
    activity->log(logger);
  }
}

const std::string MockActivityProfiler::name() const {
  return profile_name;
}

const std::set<ActivityType>& MockActivityProfiler::supportedActivityTypes() const {
  return supported_activities;
}

std::shared_ptr<IActivityProfilerSession> MockActivityProfiler::configure(
    const Config& options,
    ICompositeProfilerSession* parentSession) {
  session_ = std::make_shared<MockProfilerSession>();
  return session_;
}

std::unique_ptr<CpuTraceBuffer> MockProfilerSession::getTraceBuffer() {
  auto buf = std::make_unique<CpuTraceBuffer>();
  for (auto& i : test_activities_) {
    buf->emplace_activity(std::move(i));
  }
  test_activities_.clear();
  return buf;
}
} // namespace libkineto
