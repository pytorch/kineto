/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSession.h"

#if PTI_VERSION_AT_LEAST(0, 15)

#include "XpuptiActivityApiV2.h"

namespace KINETO_NAMESPACE {

XpuptiActivityProfilerSession::XpuptiActivityProfilerSession(
    XpuptiActivityApi& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : XpuptiActivityProfilerSessionV1(xpti, name, config, activity_types) {
  scopeProfilerEnabled_ =
      activity_types.count(ActivityType::XPU_SCOPE_PROFILER) > 0;
  if (scopeProfilerEnabled_) {
    xpti_.enableScopeProfiler(*config_);
  }
}

XpuptiActivityProfilerSession::~XpuptiActivityProfilerSession() {
  if (scopeProfilerEnabled_) {
    xpti_.disableScopeProfiler();
  }
}

void XpuptiActivityProfilerSession::start() {
  XpuptiActivityProfilerSessionV1::start();
  if (scopeProfilerEnabled_) {
    xpti_.startScopeActivity();
  }
}

void XpuptiActivityProfilerSession::stop() {
  if (scopeProfilerEnabled_) {
    xpti_.stopScopeActivity();
  }
  XpuptiActivityProfilerSessionV1::stop();
}

void XpuptiActivityProfilerSession::toggleCollectionDynamic(const bool enable) {
  XpuptiActivityProfilerSessionV1::toggleCollectionDynamic(enable);
  if (scopeProfilerEnabled_) {
    if (enable) {
      xpti_.startScopeActivity();
    } else {
      xpti_.stopScopeActivity();
    }
  }
}

void XpuptiActivityProfilerSession::processTrace(ActivityLogger& logger) {
  XpuptiActivityProfilerSessionV1::processTrace(logger);
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
