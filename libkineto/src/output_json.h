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

#ifdef HAS_CUPTI
#include <cupti.h>
#endif
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

#ifdef HAS_CUPTI
  void handleRuntimeActivity(
      const RuntimeActivity& activity) override;

  void handleGpuActivity(const GpuActivity<CUpti_ActivityKernel4>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy2>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemset>& activity) override;
#endif // HAS_CUPTI

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override;

 private:

#ifdef HAS_CUPTI
  // Create a flow event to an external event
  void handleLinkStart(const RuntimeActivity& s);
  void handleLinkEnd(const TraceActivity& e);
#endif // HAS_CUPTI

  std::string fileName_;
  std::ofstream traceOf_;

  // Cache pid to avoid repeated calls to getpid()
  pid_t pid_;

#ifdef HAS_CUPTI
  // Number of SMs on current device
  int smCount_{0};
#endif
};

} // namespace KINETO_NAMESPACE
