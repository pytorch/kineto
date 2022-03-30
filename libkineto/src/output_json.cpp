// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "output_json.h"

#include <fmt/format.h>
#include <fstream>
#include <time.h>
#include <map>

#include "Config.h"
#ifdef HAS_CUPTI
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
#include "CuptiActivityApi.h"
#include "CudaDeviceProperties.h"
#endif // HAS_CUPTI
#include "Demangle.h"
#include "TraceSpan.h"

#include "Logger.h"

using std::endl;
using namespace libkineto;

namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
static constexpr char kFlowStart = 's';
static constexpr char kFlowEnd = 'f';

#ifdef __linux__
static constexpr char kDefaultLogFileFmt[] =
    "/tmp/libkineto_activities_{}.json";
#else
static constexpr char kDefaultLogFileFmt[] = "libkineto_activities_{}.json";
#endif

std::string& ChromeTraceLogger::sanitizeStrForJSON(std::string& value) {
// Replace all backslashes with forward slash because Windows paths causing JSONDecodeError.
#ifdef _WIN32
  std::replace(value.begin(), value.end(), '\\', '/');
#endif
  return value;
}

void ChromeTraceLogger::metadataToJSON(
    const std::unordered_map<std::string, std::string>& metadata) {
  for (const auto& kv : metadata) {
    traceOf_ << fmt::format(R"JSON(
  "{}": {},)JSON", kv.first, kv.second);
  }
}

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata) {
  traceOf_ << fmt::format(R"JSON(
{{
  "schemaVersion": {},)JSON", kSchemaVersion);

#ifdef HAS_CUPTI
  traceOf_ << fmt::format(R"JSON(
  "deviceProperties": [{}
  ],)JSON", devicePropertiesJson());
#endif

  metadataToJSON(metadata);
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

void ChromeTraceLogger::handleDeviceInfo(
    const DeviceInfo& info,
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
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time, info.id,
      info.name,
      time, info.id,
      info.label,
      time, info.id,
      info.id < 8 ? info.id + 0x1000000ll : info.id);
  // clang-format on
}

void ChromeTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // thread_name needs a pid and a name arg
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}, "pid": {}, "tid": {},
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "thread_sort_index", "ph": "M", "ts": {}, "pid": {}, "tid": {},
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time, info.deviceId, info.id,
      info.name,
      time, info.deviceId, info.id,
      info.sortIndex);
  // clang-format on
}

void ChromeTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // TOOD: reserve pid = -1 for overhead but we need to rethink how to scale this for
  // other metadata
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}, "pid": -1, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}, "pid": -1, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time,
      info.name,
      time,
      0x100000All);
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
    "pid": "Spans", "tid": "{}",
    "name": "{}{} ({})",
    "args": {{
      "Op count": {}
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {},
    "pid": "Spans", "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      span.startTime, span.endTime - span.startTime,
      span.name,
      span.prefix, span.name, span.iteration,
      span.opCount,
      span.startTime,
      // Large sort index to appear at the bottom
      0x20000000ll);
  // clang-format on

  addIterationMarker(span);
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

static std::string traceActivityJson(const ITraceActivity& activity) {
  // clang-format off
  int64_t ts = activity.timestamp();
  int64_t duration = activity.duration();
  if (activity.type() ==  ActivityType::GPU_USER_ANNOTATION) {
    // The GPU user annotations start at the same time as the
    // first associated GPU activity. Since they appear later
    // in the trace file, this causes a visualization issue in Chrome.
    // Make it start one us earlier.
    ts--;
    duration++; // Still need it to end at the orginal point
  }
  return fmt::format(R"JSON(
    "name": "{}", "pid": {}, "tid": {},
    "ts": {}, "dur": {})JSON",
      activity.name(), activity.deviceId(), activity.resourceId(),
      ts, duration);
  // clang-format on
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::ITraceActivity& op) {
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
      op.timestamp(), op.metadataJson());
}

void ChromeTraceLogger::handleActivity(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  const std::string op_metadata = op.metadataJson();
  std::string separator = "";
  if (op_metadata.find_first_not_of(" \t\n") != std::string::npos) {
    separator = ",\n      ";
  }
  std::string span = "";
  if (op.traceSpan()) {
    span = fmt::format(R"JSON(
      "Trace name": "{}", "Trace iteration": {},)JSON",
        op.traceSpan()->name,
        op.traceSpan()->iteration);
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "{}", {},
    "args": {{{}
      "External id": {}{}{}
    }}
  }},)JSON",
          toString(op.type()), traceActivityJson(op),
          // args
          span,
          op.correlationId(), separator, op_metadata);
  // clang-format on
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
        handleActivity(op);
}

void ChromeTraceLogger::handleGenericLink(const ITraceActivity& act) {
  static struct {
    int type;
    char longName[24];
    char shortName[16];
  } flow_names[] = {
    {kLinkFwdBwd, "forward_backward", "fwd_bwd"},
    {kLinkAsyncCpuGpu, "async_cpu_to_gpu", "async_gpu"}
  };
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      // Link the activities via flow ID in source and destination.
      // The source node must return true from flowStart()
      // and the destination node false.
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.longName, flow.shortName);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.longName, flow.shortName);
      }
      return;
    }
  }
  LOG(WARNING) << "Unknown flow type: " << act.flowType();
}

void ChromeTraceLogger::handleLink(
    char type,
    const ITraceActivity& e,
    int64_t id,
    const std::string& cat,
    const std::string& name) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "{}", "id": {}, "pid": {}, "tid": {}, "ts": {},
    "cat": "{}", "name": "{}", "bp": "e"
  }},)JSON",
      type, id, e.deviceId(), e.resourceId(), e.timestamp(), cat, name);
  // clang-format on
}

#ifdef HAS_CUPTI
// GPU side kernel activity
void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityKernel4>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityKernel4* kernel = &activity.raw();
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
      "stream": {}, "correlation": {},
      "registers per thread": {},
      "shared memory": {},
      "blocks per SM": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "est. achieved occupancy %": {}
    }}
  }},)JSON",
      traceActivityJson(activity),
      // args
      us(kernel->queued), kernel->deviceId, kernel->contextId,
      kernel->streamId, kernel->correlationId,
      kernel->registersPerThread,
      kernel->staticSharedMemory + kernel->dynamicSharedMemory,
      blocks_per_sm,
      warps_per_sm,
      kernel->gridX, kernel->gridY, kernel->gridZ,
      kernel->blockX, kernel->blockY, kernel->blockZ,
      (int) (0.5 + occupancy * 100.0));
  // clang-format on

  auto to_id = activity.correlationId();
  handleLink(kFlowEnd, activity, to_id, "async_cpu_to_gpu", "async_gpu");
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
  VLOG(2) << memcpy.correlationId << ": MEMCPY";
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", {},
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity),
      // args
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on

  int64_t to_id = activity.correlationId();
  handleLink(kFlowEnd, activity, to_id, "async_cpu_to_gpu", "async_gpu");
}

// GPU side memcpy activity
void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityMemcpy2>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityMemcpy2& memcpy = activity.raw();
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memcpy", {},
    "args": {{
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "fromContext": {}, "inContext": {}, "toContext": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity),
      // args
      memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
      memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on

  int64_t to_id = activity.correlationId();
  handleLink(kFlowEnd, activity, to_id, "async_cpu_to_gpu", "async_gpu");
}

void ChromeTraceLogger::handleGpuActivity(
    const GpuActivity<CUpti_ActivityMemset>& activity) {
  if (!traceOf_) {
    return;
  }
  const CUpti_ActivityMemset& memset = activity.raw();
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Memset", {},
    "args": {{
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}
    }}
  }},)JSON",
      traceActivityJson(activity),
      // args
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId,
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
  // clang-format on

  int64_t to_id = activity.correlationId();
  handleLink(kFlowEnd, activity, to_id, "async_cpu_to_gpu", "async_gpu");
}
#endif // HAS_CUPTI

void ChromeTraceLogger::finalizeTrace(
    const Config& /*unused*/,
    std::unique_ptr<ActivityBuffers> /*unused*/,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  LOG(INFO) << "Chrome Trace written to " << fileName_;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": {}
  }}
  ],)JSON",
      endTime);

#if !USE_GOOGLE_LOG
  std::unordered_map<std::string, std::string> PreparedMetadata;
  for (const auto& kv : metadata) {
    // Skip empty log buckets, ex. skip ERROR if its empty.
    if (!kv.second.empty()) {
      std::string value = "[";
      // Ex. Each metadata from logger is a list of strings, expressed in JSON as
      //   "ERROR": ["Error 1", "Error 2"],
      //   "WARNING": ["Warning 1", "Warning 2", "Warning 3"],
      //   ...
      int mdv_count = kv.second.size();
      for (const auto& v : kv.second) {
        value.append("\"" + v + "\"");
        if(mdv_count > 1) {
          value.append(",");
          mdv_count--;
        }
      }
      value.append("]");
      PreparedMetadata[kv.first] = sanitizeStrForJSON(value);
    }
  }
  metadataToJSON(PreparedMetadata);
#endif // !USE_GOOGLE_LOG

  // Putting this here because the last entry MUST not end with a comma.
  traceOf_ << fmt::format(R"JSON(
  "traceName": "{}"
}})JSON", sanitizeStrForJSON(fileName_));
  // clang-format on

  traceOf_.close();
}

} // namespace KINETO_NAMESPACE
