/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_json.h"
#include "Config.h"

#include <unistd.h>

#include <fmt/format.h>
#include <time.h>
#include <map>
#include "Demangle.h"

#include "Logger.h"
#include "cupti_runtime_cbid_names.h"

using std::endl;
using namespace libkineto;

namespace KINETO_NAMESPACE {

static void openTraceFile(std::string& name, std::ofstream& stream) {
  stream.open(name, std::ofstream::out | std::ofstream::trunc);
  if (!stream) {
    PLOG(ERROR) << "Failed to open '" << name << "'";
  } else {
    LOG(INFO) << "Tracing to " << name;
    stream << "[" << endl;
  }
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName)
    : fileName_(traceFileName), pid_(getpid()) {
  traceOf_.clear(std::ios_base::badbit);
}

void ChromeTraceLogger::configure(const Config& config) {
  if (traceOf_.is_open()) {
    traceOf_.close();
  }
  fileName_ = config.activitiesLogFile();
  openTraceFile(fileName_, traceOf_);
}

static const char* getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

static const char* getMemoryKindString(CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pagable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "Managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "Device Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "Managed Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
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

void ChromeTraceLogger::handleProcessName(
    pid_t pid,
    const std::string& processName,
    const std::string& label,
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
      time, pid,
      processName,
      time, pid,
      label);
  // clang-format on
}

void ChromeTraceLogger::handleThreadName(
    uint32_t tid,
    const std::string& label,
    uint64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // process_name needs a pid and a name arg
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}, "pid": {}, "tid": {},
    "args": {{
      "name": "thread {} ({})"
    }}
  }},)JSON",
      time, pid_, tid,
      renameThreadID(tid), label);
  // clang-format on
}

void ChromeTraceLogger::handleNetCPUSpan(
    int netId,
    const std::string& netName,
    int iteration,
    int opCount,
    int gpuOpCount,
    uint64_t startTime,
    uint64_t endTime) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Net", "ts": {}, "dur": {},
    "pid": "Nets", "tid": "Net {}",
    "name": "{} CPU ({})",
    "args": {{
      "Op count": {}, "GPU op count": {}
    }}
  }},)JSON",
      startTime, endTime - startTime,
      netId,
      netName, iteration,
      opCount, gpuOpCount);
  // clang-format on
}

void ChromeTraceLogger::handleNetGPUSpan(
    int netId,
    const std::string& netName,
    int iteration,
    uint64_t startTime,
    uint64_t endTime) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Net", "ts": {}, "dur": {},
    "pid": "Nets", "tid": "Net {}",
    "name": "{} GPU ({})"
  }},)JSON",
      startTime, endTime - startTime,
      netId,
      netName, iteration);
  // clang-format on
}

void ChromeTraceLogger::handleIterationStart(
    const std::string& netName,
    int64_t time,
    uint32_t tid) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Iteration Start: {}", "ph": "i", "s": "g",
    "pid": {}, "tid": {}, "ts": {}
  }},)JSON",
      netName,
      pid_, tid, time);
  // clang-format on
}

void ChromeTraceLogger::handleCpuActivity(
    const std::string& netName,
    int netIteration,
    const external_api::OpDetails& op) {
  if (!traceOf_) {
    return;
  }

  uint64_t duration = op.endTime - op.startTime;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Operator", "ts": {}, "dur": {},
    "pid": {}, "tid": {},
    "name": "{}",
    "args": {{
       "Input dims": {}, "Input type": {}, "Input names": {},
       "Output dims": {}, "Output type": {}, "Output names": {},
       "Device": {}, "External id": {}, "Extra arguments": {},
       "Net name": "{}", "Net iteration": {}
    }}
  }},)JSON",
      op.startTime, duration,
      pid_, (uint32_t) op.threadId,
      op.opType,
      // args
      op.inputDims, op.inputTypes, op.inputNames,
      op.outputDims, op.outputTypes, op.outputNames,
      op.deviceId, op.correlationId, op.arguments,
      netName, netIteration);
  // clang-format on
}

void ChromeTraceLogger::handleLinkStart(const CUpti_ActivityAPI* activity) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "s", "id": {}, "pid": {}, "tid": {}, "ts": {},
    "cat": "async", "name": "launch"
  }},)JSON",
      activity->correlationId, pid_, activity->threadId, us(activity->start));
  // clang-format on
}

void ChromeTraceLogger::handleLinkEnd(
    uint32_t id,
    int device,
    int stream,
    uint64_t tsUsecs) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "f", "id": {}, "pid": {}, "tid": "stream {}", "ts": {},
    "cat": "async", "name": "launch", "bp": "e"
  }},)JSON",
      id, device, stream, tsUsecs);
  // clang-format on
}

void ChromeTraceLogger::handleRuntimeActivity(
    const CUpti_ActivityAPI* activity,
    const external_api::OpDetails& ext) {
  if (!traceOf_) {
    return;
  }

  uint64_t start = us(activity->start);
  if (ext.startTime == start) {
    // This will be a problem as the flow arrows will start from
    // the runtime activity rather than the external activity.
    // Adjust runtime activity start time by one to avoid this.
    ++start;
  }
  uint64_t duration = us(activity->end - activity->start);
  const char* name = runtimeCbidName(activity->cbid);
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Runtime", "ts": {}, "dur": {},
    "pid": {}, "tid": {},
    "name": "{}",
    "args": {{
      "cbid": {}, "correlation": {},
      "external id": {}, "external ts": {}
    }}
  }},)JSON",
      start, duration,
      pid_, activity->threadId,
      name,
      // args
      activity->cbid, activity->correlationId,
      ext.correlationId, ext.startTime);
  // clang-format on
}

// GPU side kernel activity
void ChromeTraceLogger::handleGpuActivity(
    const CUpti_ActivityKernel4* kernel,
    const external_api::OpDetails& ext,
    int smCount) {
  if (!traceOf_) {
    return;
  }
  uint64_t duration = us(kernel->end) - us(kernel->start);
  auto name = demangle(kernel->name);
  float warps_per_sm = (kernel->gridY * kernel->gridX * kernel->gridZ) *
      (kernel->blockX * kernel->blockY * kernel->blockZ) / 32.0 / smCount;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Kernel", "ts": {}, "dur": {},
    "pid": {}, "tid": "stream {}",
    "name": "{}",
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
      us(kernel->start), duration,
      kernel->deviceId, kernel->streamId,
      name,
      // args
      us(kernel->queued), kernel->deviceId, kernel->contextId,
      kernel->streamId, kernel->correlationId, ext.correlationId,
      kernel->registersPerThread,
      kernel->staticSharedMemory + kernel->dynamicSharedMemory,
      warps_per_sm,
      kernel->gridX, kernel->gridY, kernel->gridZ,
      kernel->blockX, kernel->blockY, kernel->blockZ);
  // clang-format on
}

// GPU side memcpy activity
void ChromeTraceLogger::handleGpuActivity(
    const CUpti_ActivityMemcpy* memcpy,
    const external_api::OpDetails& ext) {
  if (!traceOf_) {
    return;
  }
  const char* copy_kind =
      getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind);
  const char* src_kind =
      getMemoryKindString((CUpti_ActivityMemoryKind)memcpy->srcKind);
  const char* dst_kind =
      getMemoryKindString((CUpti_ActivityMemoryKind)memcpy->dstKind);
  uint64_t duration = us(memcpy->end) - us(memcpy->start);
  VLOG(2) << memcpy->correlationId << ": MEMCPY";
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", "ts": {}, "dur": {},
    "pid": {}, "tid": "stream {}",
    "name": "Memcpy {} ({} -> {})",
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      us(memcpy->start), duration,
      memcpy->deviceId, memcpy->streamId,
      copy_kind, src_kind, dst_kind,
      // args
      memcpy->deviceId, memcpy->contextId,
      memcpy->streamId, memcpy->correlationId, ext.correlationId,
      memcpy->bytes, memcpy->bytes * 1.0 / (memcpy->end - memcpy->start));
  // clang-format on
}

// GPU side memcpy activity
void ChromeTraceLogger::handleGpuActivity(
    const CUpti_ActivityMemcpy2* memcpy,
    const external_api::OpDetails& ext) {
  if (!traceOf_) {
    return;
  }
  const char* copy_kind =
      getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind);
  const char* src_kind =
      getMemoryKindString((CUpti_ActivityMemoryKind)memcpy->srcKind);
  const char* dst_kind =
      getMemoryKindString((CUpti_ActivityMemoryKind)memcpy->dstKind);
  uint64_t duration = us(memcpy->end) - us(memcpy->start);
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", "ts": {}, "dur": {},
    "pid": {}, "tid": "stream {}",
    "name": "Memcpy {} ({} -> {})",
    "args": {{
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "fromContext": {}, "inContext": {}, "toContext": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      us(memcpy->start), duration,
      memcpy->deviceId, memcpy->streamId,
      copy_kind, src_kind, dst_kind,
      // args
      memcpy->srcDeviceId, memcpy->deviceId, memcpy->dstDeviceId,
      memcpy->srcContextId, memcpy->contextId, memcpy->dstContextId,
      memcpy->streamId, memcpy->correlationId, ext.correlationId,
      memcpy->bytes, memcpy->bytes * 1.0 / (memcpy->end - memcpy->start));
  // clang-format on
}

void ChromeTraceLogger::handleGpuActivity(
    const CUpti_ActivityMemset* memset,
    const external_api::OpDetails& ext) {
  if (!traceOf_) {
    return;
  }
  auto memory_kind =
      getMemoryKindString((CUpti_ActivityMemoryKind)memset->memoryKind);
  uint64_t duration = us(memset->end) - us(memset->start);
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memset", "ts": {}, "dur": {},
    "pid": {}, "tid": "stream {}",
    "name": "Memset ({})",
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {}, "external id": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      us(memset->start), duration,
      memset->deviceId, memset->streamId,
      memory_kind,
      // args
      memset->deviceId, memset->contextId,
      memset->streamId, memset->correlationId, ext.correlationId,
      memset->bytes, memset->bytes * 1.0 / (memset->end - memset->start));
  // clang-format on
}

void ChromeTraceLogger::finalizeTrace(const Config& config) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  traceOf_.close();
  LOG(INFO) << "Chrome Trace written to " << config.activitiesLogFile();
}

} // namespace KINETO_NAMESPACE
