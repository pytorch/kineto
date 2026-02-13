/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSessionV2.h"

#if PTI_VERSION_AT_LEAST(0, 15)

#include "XpuptiActivityApiV2.h"

namespace KINETO_NAMESPACE {

XpuptiActivityProfilerSessionV2::XpuptiActivityProfilerSessionV2(
    XpuptiActivityApiV2& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : XpuptiActivityProfilerSession(xpti, name, config, activity_types) {
  scopeProfilerEnabled_ =
      activity_types.count(ActivityType::XPU_SCOPE_PROFILER) > 0;
  if (scopeProfilerEnabled_) {
    xpti_.enableScopeProfiler(*config_);
  }
}

XpuptiActivityProfilerSessionV2::~XpuptiActivityProfilerSessionV2() {
  if (scopeProfilerEnabled_) {
    xpti_.disableScopeProfiler();
  }
}

void XpuptiActivityProfilerSessionV2::start() {
  XpuptiActivityProfilerSession::start();
  if (scopeProfilerEnabled_) {
    xpti_.startScopeActivity();
  }
}

void XpuptiActivityProfilerSessionV2::stop() {
  if (scopeProfilerEnabled_) {
    xpti_.stopScopeActivity();
  }
  XpuptiActivityProfilerSession::stop();
}

void XpuptiActivityProfilerSessionV2::toggleCollectionDynamic(
    const bool enable) {
  XpuptiActivityProfilerSession::toggleCollectionDynamic(enable);
  if (scopeProfilerEnabled_) {
    if (enable) {
      xpti_.startScopeActivity();
    } else {
      xpti_.stopScopeActivity();
    }
  }
}

void XpuptiActivityProfilerSessionV2::processTrace(ActivityLogger& logger) {
  XpuptiActivityProfilerSession::processTrace(logger);
  if (scopeProfilerEnabled_) {
    xpti_.processScopeTrace(
        [this, &logger](
            const pti_metrics_scope_record_t* record,
            const pti_metrics_scope_record_metadata_t& metadata) -> void {
          handleScopeRecord(record, metadata, logger);
        });
  }
}

} // namespace KINETO_NAMESPACE

#endif
