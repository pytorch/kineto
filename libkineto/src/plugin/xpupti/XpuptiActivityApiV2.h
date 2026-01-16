/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiActivityApi.h"

#if PTI_VERSION_AT_LEAST(0, 15)

#include <pti/pti_metrics_scope.h>

#include <optional>

namespace KINETO_NAMESPACE {

class Config;

class XpuptiActivityApi : public XpuptiActivityApiV1 {
 public:
  XpuptiActivityApi() = default;
  XpuptiActivityApi(const XpuptiActivityApi&) = delete;
  XpuptiActivityApi& operator=(const XpuptiActivityApi&) = delete;

  virtual ~XpuptiActivityApi() {}

  static XpuptiActivityApi& singleton();

  void enableXpuptiActivities(
      const std::set<ActivityType>& selected_activities) {
    return XpuptiActivityApiV1::enableXpuptiActivities(
        selected_activities, true);
  }

  void enableScopeProfiler(const Config&);
  void disableScopeProfiler();
  void startScopeActivity();
  void stopScopeActivity();

  void processScopeTrace(
      std::function<void(
          const pti_metrics_scope_record_t*,
          const pti_metrics_scope_record_metadata_t& metadata)> handler);

 private:
  struct safe_pti_scope_collection_handle_t {
    safe_pti_scope_collection_handle_t(
        std::exception_ptr& exceptFromDestructor);
    ~safe_pti_scope_collection_handle_t() noexcept;

    operator pti_scope_collection_handle_t() {
      return handle_;
    }

    pti_scope_collection_handle_t handle_{};
    std::exception_ptr& exceptFromDestructor_;
  };

  std::optional<safe_pti_scope_collection_handle_t> scopeHandleOpt_;
  std::exception_ptr exceptFromScopeHandleDestructor_;
};

} // namespace KINETO_NAMESPACE

#else

namespace KINETO_NAMESPACE {

struct XpuptiActivityApi : public XpuptiActivityApiV1 {
  using XpuptiActivityApiV1::XpuptiActivityApiV1;

  static XpuptiActivityApi& singleton();
};

} // namespace KINETO_NAMESPACE

#endif
