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
#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  // Previous declaration of TraceSpan is struct. Must match the same here.
  struct TraceSpan;
}

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleProcessInfo(
      const ProcessInfo& processInfo,
      uint64_t time) override;

  void handleThreadInfo(const ThreadInfo& threadInfo, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleGenericActivity(const GenericTraceActivity& activity) override;

#ifdef HAS_CUPTI
  void handleRuntimeActivity(
      const RuntimeActivity& activity) override;

  void handleGpuActivity(const GpuActivity<CUpti_ActivityKernel4>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy2>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemset>& activity) override;
#endif // HAS_CUPTI

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override;

  std::string traceFileName() const {
    return fileName_;
  }

 private:

#ifdef HAS_CUPTI
  // Create a flow event to an external event
  void handleLinkStart(const RuntimeActivity& s);
  void handleLinkEnd(const TraceActivity& e);
#endif // HAS_CUPTI

  void addIterationMarker(const TraceSpan& span);

  void openTraceFile();

  void handleGenericInstantEvent(const GenericTraceActivity& op);

  void handleGenericLink(const GenericTraceActivity& activity);
  void handleFwdBwdLinkStart(const GenericTraceActivity& s, const std::string& cat, const std::string& name);
  void handleFwdBwdLinkEnd(const GenericTraceActivity& e, const std::string& cat, const std::string& name);

  std::string fileName_;
  std::ofstream traceOf_;
};

} // namespace KINETO_NAMESPACE
