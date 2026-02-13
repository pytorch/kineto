/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiActivityProfilerSessionV1.h"

#if PTI_VERSION_AT_LEAST(0, 15)

#include <pti/pti_metrics_scope.h>

namespace KINETO_NAMESPACE {

class XpuptiActivityProfilerSession : public XpuptiActivityProfilerSessionV1 {
 public:
  XpuptiActivityProfilerSession(
      XpuptiActivityApi& xpti,
      const std::string& name,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);

  XpuptiActivityProfilerSession(const XpuptiActivityProfilerSession&) = delete;
  XpuptiActivityProfilerSession& operator=(
      const XpuptiActivityProfilerSession&) = delete;

  ~XpuptiActivityProfilerSession();
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

#else

namespace KINETO_NAMESPACE {
using XpuptiActivityProfilerSession = XpuptiActivityProfilerSessionV1;
} // namespace KINETO_NAMESPACE

#endif
