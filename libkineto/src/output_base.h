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
#include "ActivityBuffers.h"
#include "ClientTraceActivity.h"
#include "CuptiActivity.h"
#include "ProcessInfo.h"
#include "TraceSpan.h"

namespace KINETO_NAMESPACE {
  class Config;
  class RuntimeActivity;
  class GpuKernelActivity;
}

namespace libkineto {

using namespace KINETO_NAMESPACE;

class ActivityLogger {
 public:
  virtual ~ActivityLogger() = default;

  virtual void handleProcessInfo(
      const ProcessInfo& processInfo,
      uint64_t time) = 0;

  virtual void handleThreadInfo(const ThreadInfo& threadInfo, int64_t time) = 0;

  virtual void handleTraceSpan(const TraceSpan& span) = 0;

  virtual void handleIterationStart(const TraceSpan& span) = 0;

  virtual void handleCpuActivity(
      const libkineto::ClientTraceActivity& activity,
      const TraceSpan& span) = 0;

  virtual void handleRuntimeActivity(const RuntimeActivity& activity) = 0;

  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityKernel4>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemcpy>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemcpy2>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemset>& activity) = 0;

  virtual void finalizeTrace(
      const KINETO_NAMESPACE::Config& config,
      std::unique_ptr<ActivityBuffers> buffers) = 0;

 protected:
  ActivityLogger() = default;

  // get a cleaner thread id
  int renameThreadID(size_t tid) {
    // the tid here is the thread ID that schedules the operator
    static int curr_tid = 0;

    // Note this function is not thread safe; The user of this
    // ActivityLogger need to maintain thread safety
    if (tidMap_.count(tid)) {
      return tidMap_[tid];
    } else {
      return tidMap_[tid] = curr_tid++;
    }
  }

 private:
  // store the mapping of thread id vs. showing on the trace
  std::unordered_map<size_t, int> tidMap_;
};

} // namespace KINETO_NAMESPACE
