/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_json.h"

#include <fmt/format.h>
#include <fstream>
#include <time.h>
#include <map>

#include "Config.h"
#ifdef HAS_CUPTI
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
#include "CuptiActivityInterface.h"
#include "CudaDeviceProperties.h"
#endif // HAS_CUPTI
#include "Demangle.h"
#include "TraceSpan.h"

#include "Logger.h"

using std::endl;
using namespace libkineto;

namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
#ifdef __linux__
static const std::string kDefaultLogFileFmt =
    "/tmp/libkineto_activities_{}.json";
#else
static const std::string kDefaultLogFileFmt = "libkineto_activities_{}.json";
#endif

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata) {
  traceOf_ << fmt::format(R"JSON(
{{
  "schemaVersion": {},)JSON", kSchemaVersion);

  for (const auto& kv : metadata) {
    traceOf_ << fmt::format(R"JSON(
  "{}": {},)JSON", kv.first, kv.second);
  }

#ifdef HAS_CUPTI
  traceOf_ << fmt::format(R"JSON(
  "deviceProperties": [{}
  ],)JSON", devicePropertiesJson());
#endif

  traceOf_ << R"JSON(
  "traceEvents": [)JSON";
}

static std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

void ChromeTraceLogger::openTraceFile() {
  traceOf_.open(fileName_, std::ofstream::out | std::ofstream::trunc);
  if (!traceOf_) {
    PLOG(ERROR) << "Failed to open '" << fileName_ << "'";
  } else {
    LOG(INFO) << "Tracing to " << fileName_;
  }
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName) {
  fileName_ = traceFileName.empty() ? defaultFileName() : traceFileName;
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile();
}

static int64_t us(int64_t timestamp) {
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
      time, processId(), threadInfo.tid,
      threadInfo.tid, threadInfo.name);
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

  if (span.tracked) {
    addIterationMarker(span);
  }
}

void ChromeTraceLogger::addIterationMarker(const TraceSpan& span) {
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

static std::string traceActivityJson(
    const TraceActivity& activity, std::string tid) {
  // clang-format off
  return fmt::format(R"JSON(
    "name": "{}", "pid": {}, "tid": "{}",
    "ts": {}, "dur": {})JSON",
      activity.name(), activity.deviceId(), tid,
      activity.timestamp(), activity.duration());
  // clang-format on
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::GenericTraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "i", "s": "t", "name": "{}",
    "pid": {}, "tid": {},
    "ts": {},
    "args": {{
      {}
    }}
  }},)JSON",
      op.name(), op.deviceId(), op.resourceId(),
      op.timestamp(), op.getMetadata());
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.activityType == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  auto op_metadata = op.getMetadata();
  std::string separator = "";
  if (op_metadata.find_first_not_of(" \t\n") != std::string::npos) {
    separator = ",";
  }
  const std::string tid =
      op.type() == ActivityType::GPU_USER_ANNOTATION ?
      fmt::format("stream {} annotations", op.resourceId()) :
      fmt::format("{}", op.resourceId());

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "{}", {},
    "args": {{
       "External id": {},
       "Trace name": "{}", "Trace iteration": {}{}
       {}
    }}
  }},)JSON",
      toString(op.type()), traceActivityJson(op, tid),
      // args
      op.id,
      op.traceSpan()->name, op.traceSpan()->iteration, separator,
      op_metadata);
  // clang-format on
}

#ifdef HAS_CUPTI
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
      s.correlationId(), processId(), s.resourceId(), s.timestamp());
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
      traceActivityJson(activity, fmt::format("{}", activity.resourceId())),
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

static std::string streamName(const TraceActivity& act) {
  return fmt::format("stream {}", act.resourceId());
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
  float blocks_per_sm = -1.0;
  float warps_per_sm = -1.0;
  int sm_count = smCount(kernel->deviceId);
  if (sm_count) {
    blocks_per_sm =
        (kernel->gridX * kernel->gridY * kernel->gridZ) / (float) sm_count;
    warps_per_sm =
        blocks_per_sm * (kernel->blockX * kernel->blockY * kernel->blockZ)
        / threads_per_warp;
  }

  // Calculate occupancy
  float occupancy = KINETO_NAMESPACE::kernelOccupancy(
      kernel->deviceId,
      kernel->registersPerThread,
      kernel->staticSharedMemory,
      kernel->dynamicSharedMemory,
      kernel->blockX,
      kernel->blockY,
      kernel->blockZ,
      blocks_per_sm);

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Kernel", {},
    "args": {{
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "registers per thread": {},
      "shared memory": {},
      "blocks per SM": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "est. achieved occupancy %": {}
    }}
  }},)JSON",
      traceActivityJson(activity, streamName(activity)),
      // args
      us(kernel->queued), kernel->deviceId, kernel->contextId,
      kernel->streamId, kernel->correlationId, ext.correlationId(),
      kernel->registersPerThread,
      kernel->staticSharedMemory + kernel->dynamicSharedMemory,
      blocks_per_sm,
      warps_per_sm,
      kernel->gridX, kernel->gridY, kernel->gridZ,
      kernel->blockX, kernel->blockY, kernel->blockZ,
      (int) (0.5 + occupancy * 100.0));
  // clang-format on

  handleLinkEnd(activity);
}

static std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
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
      traceActivityJson(activity, streamName(activity)),
      // args
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId, ext.correlationId(),
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
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
      traceActivityJson(activity, streamName(activity)),
      // args
      memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
      memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
      memcpy.streamId, memcpy.correlationId, ext.correlationId(),
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
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
      traceActivityJson(activity, streamName(activity)),
      // args
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId, ext.correlationId(),
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
  // clang-format on

  handleLinkEnd(activity);
}
#endif // HAS_CUPTI

void ChromeTraceLogger::finalizeTrace(
    const Config& /*unused*/,
    std::unique_ptr<ActivityBuffers> /*unused*/,
    int64_t endTime) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": {}
  }}
]}})JSON",
      endTime);
  // clang-format on

  traceOf_.close();
  LOG(INFO) << "Chrome Trace written to " << fileName_;
}

} // namespace KINETO_NAMESPACE
