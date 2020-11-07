/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_json.h"

#include <fmt/format.h>
#include <time.h>
#include <map>
#include <unistd.h>

#include "cupti_strings.h"
#include "Config.h"
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
#include "CuptiActivityInterface.h"
#include "Demangle.h"
#include "TraceSpan.h"

#include "Logger.h"

using std::endl;
using namespace libkineto;

namespace KINETO_NAMESPACE {

static void openTraceFile(std::string& name, std::ofstream& stream) {
  stream.open(name, std::ofstream::out | std::ofstream::trunc);
  if (!stream) {
    PLOG(ERROR) << "Failed to open '" << name << "'";
  } else {
    LOG(INFO) << "Logging to " << name;
    stream << "[" << endl;
  }
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName)
    : fileName_(traceFileName), pid_(getpid()) {
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile(fileName_, traceOf_);
  smCount_ = CuptiActivityInterface::singleton().smCount();
}

int ChromeTraceLogger::renameThreadID(uint32_t tid) {
  // the tid here is the thread ID that schedules the operator
  static int curr_tid = 0;

  // Note this function is not thread safe; The user of this ChromeTraceLogger
  // need to maintain thread safety
  if (tidMap_.count(tid)) {
    return tidMap_[tid];
  } else {
    return tidMap_[tid] = curr_tid++;
  }
}

static uint64_t us(uint64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

void ChromeTraceLogger::handleProcessInfo(
    const ProcessInfo& processInfo,
    uint64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // process_name needs a pid and a name arg
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_labels", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "labels": "{}"
    }}
  }},)JSON",
      time, processInfo.pid,
      processInfo.name,
      time, processInfo.pid,
      processInfo.label);
  // clang-format on
}

void ChromeTraceLogger::handleThreadInfo(
    const ThreadInfo& threadInfo,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // thread_name needs a pid and a name arg
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}, "pid": {}, "tid": "{}",
    "args": {{
      "name": "thread {} ({})"
    }}
  }},)JSON",
      time, pid_, (uint32_t)threadInfo.tid,
      renameThreadID((uint32_t)threadInfo.tid), threadInfo.name);
  // clang-format on
}

void ChromeTraceLogger::handleTraceSpan(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Trace", "ts": {}, "dur": {},
    "pid": "Traces", "tid": "{}",
    "name": "{}{} ({})",
    "args": {{
      "Op count": {}
    }}
  }},)JSON",
      span.startTime, span.endTime - span.startTime,
      span.name,
      span.prefix, span.name, span.iteration,
      span.opCount);
  // clang-format on
}

void ChromeTraceLogger::handleIterationStart(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Iteration Start: {}", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace {}", "ts": {}
  }},)JSON",
      span.name,
      span.name, span.startTime);
  // clang-format on
}

static std::string traceActivityJson(const TraceActivity& activity, std::string tidPrefix) {
  // clang-format off
  return fmt::format(R"JSON(
    "name": "{}", "pid": {}, "tid": "{}{}",
    "ts": {}, "dur": {})JSON",
      activity.name(), activity.deviceId(), tidPrefix, (uint32_t)activity.resourceId(),
      activity.timestamp(), activity.duration());
  // clang-format on
}

void ChromeTraceLogger::handleCpuActivity(
    const libkineto::ClientTraceActivity& op,
    const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  uint64_t duration = op.endTime - op.startTime;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Operator", {},
    "args": {{
       "Input dims": {}, "Input type": {}, "Input names": {},
       "Output dims": {}, "Output type": {}, "Output names": {},
       "Device": {}, "External id": {}, "Extra arguments": {},
       "Trace name": "{}", "Trace iteration": {}
    }}
  }},)JSON",
      traceActivityJson(op, ""),
      // args
      op.inputDims, op.inputTypes, op.inputNames,
      op.outputDims, op.outputTypes, op.outputNames,
      op.device, op.correlation, op.arguments,
      span.name, span.iteration);
  // clang-format on
}

void ChromeTraceLogger::handleLinkStart(const RuntimeActivity& s) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "s", "id": {}, "pid": {}, "tid": {}, "ts": {},
    "cat": "async", "name": "launch"
  }},)JSON",
      s.correlationId(), pid_, s.resourceId(), s.timestamp());
  // clang-format on

}

void ChromeTraceLogger::handleLinkEnd(const TraceActivity& e) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "f", "id": {}, "pid": {}, "tid": "stream {}", "ts": {},
    "cat": "async", "name": "launch", "bp": "e"
  }},)JSON",
      e.correlationId(), e.deviceId(), e.resourceId(), e.timestamp());
  // clang-format on
}

void ChromeTraceLogger::handleRuntimeActivity(
    const RuntimeActivity& activity) {
  if (!traceOf_) {
    return;
  }

  const CUpti_CallbackId cbid = activity.raw().cbid;
  const TraceActivity& ext = *activity.linkedActivity();
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Runtime", {},
    "args": {{
      "cbid": {}, "correlation": {},
      "external id": {}, "external ts": {}
    }}
  }},)JSON",
      traceActivityJson(activity, ""),
      // args
      cbid, activity.raw().correlationId,
      ext.correlationId(), ext.timestamp());
  // clang-format on

  // FIXME: This is pretty hacky and it's likely that we miss some links.
  // May need to maintain a map instead.
  if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
      (cbid >= CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
       cbid <= CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020) ||
      cbid ==
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
      cbid ==
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000) {
    handleLinkStart(activity);
  }
}

// GPU side kernel activity
void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityKernel4>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityKernel4* kernel = &activity.raw();
  const TraceActivity& ext = *activity.linkedActivity();
  constexpr int threads_per_warp = 32;
  float warps_per_sm = (kernel->gridX * kernel->gridY * kernel->gridZ) *
      (kernel->blockX * kernel->blockY * kernel->blockZ) / (float) threads_per_warp / smCount_;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Kernel", {},
    "args": {{
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "registers per thread": {},
      "shared memory": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}]
    }}
  }},)JSON",
      traceActivityJson(activity, "stream "),
      // args
      us(kernel->queued), kernel->deviceId, kernel->contextId,
      kernel->streamId, kernel->correlationId, ext.correlationId(),
      kernel->registersPerThread,
      kernel->staticSharedMemory + kernel->dynamicSharedMemory,
      warps_per_sm,
      kernel->gridX, kernel->gridY, kernel->gridZ,
      kernel->blockX, kernel->blockY, kernel->blockZ);
  // clang-format on

  handleLinkEnd(activity);
}

// GPU side memcpy activity
void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityMemcpy>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityMemcpy& memcpy = activity.raw();
  const TraceActivity& ext = *activity.linkedActivity();
  VLOG(2) << memcpy.correlationId << ": MEMCPY";
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", {},
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity, "stream "),
      // args
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId, ext.correlationId(),
      memcpy.bytes, memcpy.bytes * 1.0 / (memcpy.end - memcpy.start));
  // clang-format on

  handleLinkEnd(activity);
}

// GPU side memcpy activity
void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityMemcpy2>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityMemcpy2& memcpy = activity.raw();
  const TraceActivity& ext = *activity.linkedActivity();
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", {},
    "args": {{
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "fromContext": {}, "inContext": {}, "toContext": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity, "stream "),
      // args
      memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
      memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
      memcpy.streamId, memcpy.correlationId, ext.correlationId(),
      memcpy.bytes, memcpy.bytes * 1.0 / (memcpy.end - memcpy.start));
  // clang-format on

  handleLinkEnd(activity);
}

void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityMemset>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityMemset& memset = activity.raw();
  const TraceActivity& ext = *activity.linkedActivity();
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memset", {},
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity, "stream "),
      // args
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId, ext.correlationId(),
      memset.bytes, memset.bytes * 1.0 / (memset.end - memset.start));
  // clang-format on

  handleLinkEnd(activity);
}

void ChromeTraceLogger::finalizeTrace(
    const Config& config, std::unique_ptr<ActivityBuffers> /*unused*/) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  traceOf_.close();
  LOG(INFO) << "Chrome Trace written to " << config.activitiesLogFile();
}

} // namespace KINETO_NAMESPACE
