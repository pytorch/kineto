/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfiler.h"
#include "XpuptiActivityApiV2.h"
#include "XpuptiActivityProfilerSession.h"

namespace KINETO_NAMESPACE {

[[noreturn]] const std::set<ActivityType>&
XPUActivityProfiler::availableActivities() const {
  throw std::runtime_error(
      "The availableActivities is legacy method and should not be called by kineto");
}

std::unique_ptr<libkineto::IActivityProfilerSession>
XPUActivityProfiler::configure(
    const std::set<ActivityType>& activity_types,
    const libkineto::Config& config) {
  return std::make_unique<XpuptiActivityProfilerSession>(
      XpuptiActivityApi::singleton(), name(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
XPUActivityProfiler::configure(
    [[maybe_unused]] int64_t ts_ms,
    [[maybe_unused]] int64_t duration_ms,
    const std::set<ActivityType>& activity_types,
    const libkineto::Config& config) {
  return configure(activity_types, config);
}

} // namespace KINETO_NAMESPACE
