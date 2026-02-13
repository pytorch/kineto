/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiActivityProfilerSession.h"

#if PTI_VERSION_AT_LEAST(0, 15)

#include <pti/pti_metrics_scope.h>

namespace KINETO_NAMESPACE {

class XpuptiActivityApiV2;

class XpuptiActivityProfilerSessionV2 : public XpuptiActivityProfilerSession {
 public:
  XpuptiActivityProfilerSessionV2(
      XpuptiActivityApiV2& xpti,
      const std::string& name,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);

  XpuptiActivityProfilerSessionV2(const XpuptiActivityProfilerSessionV2&) =
      delete;
  XpuptiActivityProfilerSessionV2& operator=(
      const XpuptiActivityProfilerSessionV2&) = delete;

  ~XpuptiActivityProfilerSessionV2();
  void start();
  void stop();
  void toggleCollectionDynamic(const bool enable);

  void processTrace(ActivityLogger& logger) override;

  void handleScopeRecord(
      const pti_metrics_scope_record_t* record,
      const pti_metrics_scope_record_metadata_t& metadata,
      ActivityLogger& logger);

 private:
  bool scopeProfilerEnabled_{false};
};

} // namespace KINETO_NAMESPACE

#endif
