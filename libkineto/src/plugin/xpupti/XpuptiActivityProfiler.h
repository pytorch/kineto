/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiProfilerMacros.h"

#include "IActivityProfiler.h"
#include "libkineto.h"

#include <pti/pti_view.h>

#if PTI_VERSION_AT_LEAST(0, 14)
#include <pti/pti_metrics_scope.h>
#endif

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KINETO_NAMESPACE {

using DeviceUUIDsT = std::array<unsigned char, 16>;

class XpuptiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  XpuptiActivityProfilerSession() = delete;
  XpuptiActivityProfilerSession(
      XpuptiActivityApi& xpti,
      const std::string& name,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);
  XpuptiActivityProfilerSession(const XpuptiActivityProfilerSession&) = delete;
  XpuptiActivityProfilerSession& operator=(
      const XpuptiActivityProfilerSession&) = delete;

  ~XpuptiActivityProfilerSession();

  void start() override;
  void stop() override;
  void toggleCollectionDynamic(const bool);
  std::vector<std::string> errors() override {
    return errors_;
  };
  void processTrace(ActivityLogger& logger) override;
  void processTrace(
      ActivityLogger& logger,
      libkineto::getLinkedActivityCallback get_linked_activity,
      int64_t captureWindowStartTime,
      int64_t captureWindowEndTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return {};
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

 private:
  void checkTimestampOrder(const ITraceActivity* act1);
  void removeCorrelatedPtiActivities(const ITraceActivity* act1);
  bool outOfScope(const ITraceActivity* act);
  int64_t getMappedQueueId(uint64_t sycl_queue_id);
  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);
  void handleCorrelationActivity(
      const pti_view_record_external_correlation* correlation);

#if PTI_VERSION_AT_LEAST(0, 11)
  using pti_view_record_api_t = pti_view_record_api;
#else
  using pti_view_record_api_t = pti_view_record_sycl_runtime;
#endif

  std::string getApiName(const pti_view_record_api_t* activity);

  template <class pti_view_memory_record_type>
  void handleRuntimeKernelMemcpyMemsetActivities(
      const pti_view_memory_record_type* activity,
      ActivityLogger& logger);

  void handleOverheadActivity(
      const pti_view_record_overhead* activity,
      ActivityLogger& logger);
  void handlePtiActivity(
      const pti_view_record_base* record,
      ActivityLogger& logger);

  void handleScopeRecord(
      const pti_metrics_scope_record_t* record,
      const pti_metric_scope_display_info_t* displayInfo,
      uint32_t infoCount,
      ActivityLogger& logger);

  // enumerate XPU Device UUIDs from runtime for once
  void enumDeviceUUIDs();

  // get logical device index(int8) from the given UUID from runtime
  // for profiling activity creation
  DeviceIndex_t getDeviceIdxFromUUID(const uint8_t deviceUUID[16]);

 private:
  static uint32_t iterationCount_;
  static std::vector<DeviceUUIDsT> deviceUUIDs_;
  static std::unordered_set<std::string_view> correlateRuntimeOps_;

  int64_t captureWindowStartTime_{0};
  int64_t captureWindowEndTime_{0};
  int64_t profilerStartTs_{0};
  int64_t profilerEndTs_{0};
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;
  std::unordered_map<int64_t, const ITraceActivity*> correlatedPtiActivities_;
  std::vector<std::string> errors_;

  libkineto::getLinkedActivityCallback cpuActivity_;

  XpuptiActivityApi& xpti_;
  libkineto::CpuTraceBuffer traceBuffer_;
  std::unordered_map<uint64_t, uint64_t> sycl_queue_pool_;
  std::unique_ptr<const libkineto::Config> config_{nullptr};
  const std::set<ActivityType>& activity_types_;
  std::string name_;
  bool scopeProfilerEnabled_{false};
};

class XPUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  XPUActivityProfiler() = default;
  XPUActivityProfiler(const XPUActivityProfiler&) = delete;
  XPUActivityProfiler& operator=(const XPUActivityProfiler&) = delete;

  const std::string& name() const override {
    return name_;
  }

  const std::set<ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"__xpu_profiler__"};
};

} // namespace KINETO_NAMESPACE
