/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#include <cupti.h>
#include "ClientTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  class TraceSpan;
}

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName, int smCount);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleProcessInfo(
      const ProcessInfo& processInfo,
      uint64_t time) override;

  void handleThreadInfo(const ThreadInfo& threadInfo, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleIterationStart(const TraceSpan& span) override;

  void handleCpuActivity(
      const libkineto::ClientTraceActivity& activity,
      const TraceSpan& span) override;

  void handleGenericActivity(
      const GenericTraceActivity& activity) override;

  void handleRuntimeActivity(
      const RuntimeActivity& activity) override;

  void handleGpuActivity(const GpuActivity<CUpti_ActivityKernel4>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy2>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemset>& activity) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override;

 private:
  // Create a flow event to an external event
  void handleLinkStart(const RuntimeActivity& s);
  void handleLinkEnd(const TraceActivity& e);

  void logActivity(const CUpti_Activity* act);

  std::string fileName_;
  std::ofstream traceOf_;

  // store the mapping of thread id vs. showing on the trace
  std::unordered_map<uint32_t, int> tidMap_;

  // get a cleaner thread id
  int renameThreadID(uint32_t tid);

  // Cache pid to avoid repeated calls to getpid()
  pid_t pid_;

  // Number of SMs on current device
  int smCount_;
};

} // namespace KINETO_NAMESPACE
