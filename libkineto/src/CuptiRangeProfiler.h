/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>

#include <libkineto.h>
#include <IActivityProfiler.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiRangeProfilerApi.h"

/* CuptiRangeProfiler :
 *   This profiler object provides an interface to run the CUPTI
 *   Range Based Profiler API.
 */

namespace KINETO_NAMESPACE {

using CuptiProfilerPrePostCallback = std::function<void(void)>;

/* Activity Profiler session encapsulates the CUPTI Range based Profiler
 * API object
 */
class CuptiRangeProfilerSession : public IActivityProfilerSession {
 public:
  explicit CuptiRangeProfilerSession(
      const Config& config,
      ICuptiRBProfilerSessionFactory& factory);

  ~CuptiRangeProfilerSession() override {};

  // start profiling
  void start() override;

  // stop profiling
  void stop() override;

  // process trace events with logger
  void processTrace(libkineto::ActivityLogger& logger) override;

  std::unique_ptr<CpuTraceBuffer> getTraceBuffer() override {
    return std::make_unique<CpuTraceBuffer>(std::move(traceBuffer_));
  }

  // returns errors with this trace
  std::vector<std::string> errors() override;

  // returns device info used in this trace, could be nullptr
  std::unique_ptr<DeviceInfo> getDeviceInfo() override;

  // returns resource info used in this trace, could be empty
  std::vector<ResourceInfo> getResourceInfos() override;

 private:
  void addRangeEvents(
      const CuptiProfilerResult& result,
      const CuptiRBProfilerSession* profiler);

  CUpti_ProfilerRange rangeType_ = CUPTI_UserRange;
  CUpti_ProfilerReplayMode replayType_ = CUPTI_UserReplay;

  CpuTraceBuffer traceBuffer_;
  std::vector<
    std::unique_ptr<CuptiRBProfilerSession>> profilers_;
};


/* This is a wrapper class that refers to the underlying
 * CuptiRangeProfiler. Using a wrapper libkineto can manage the ownership
 * of this object independent of the CuptiRangeProfiler itself.
 */
class CuptiRangeProfiler : public libkineto::IActivityProfiler {
 public:
  explicit CuptiRangeProfiler();

  explicit CuptiRangeProfiler(ICuptiRBProfilerSessionFactory& factory);

  ~CuptiRangeProfiler() override {}

  // name of profiler
  const std::string& name() const override;

  // returns activity types this profiler supports
  const std::set<ActivityType>& availableActivities() const override;

  // sets up the tracing session and provides control to the
  // the activity profiler session object.
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const Config& config) override;

  // asynchronous version of the above with future timestamp and duration.
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const Config& config) override;

  // hooks to enable configuring the environment before and after the
  // profiling sesssion.
  static void setPreRunCallback(CuptiProfilerPrePostCallback fn);
  static void setPostRunCallback(CuptiProfilerPrePostCallback fn);
 private:
  ICuptiRBProfilerSessionFactory& factory_;
};

struct CuptiRangeProfilerInit {
  CuptiRangeProfilerInit();
  bool success = false;
};

} // namespace KINETO_NAMESPACE
