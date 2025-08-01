/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_json.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <ctime>
#include <fstream>
#include <iterator>
#include "Config.h"
#include "TraceSpan.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
static constexpr char kFlowStart = 's';
static constexpr char kFlowEnd = 'f';

// CPU op name that is used to store collectives metadata
// TODO: share the same string across c10d, profiler and libkineto
static constexpr const char* kParamCommsCallName = "record_param_comms";
// Collective function metadata populated from CPU op to GPU kernel
static constexpr const char* kCollectiveName = "Collective name";
static constexpr const char* kDtype = "dtype";
static constexpr const char* kInMsgNelems = "In msg nelems";
static constexpr const char* kOutMsgNelems = "Out msg nelems";
static constexpr const char* kGroupSize = "Group size";
static constexpr const char* kInSplit = "In split size";
static constexpr const char* kOutSplit = "Out split size";
static constexpr const char* kProcessGroupName = "Process Group Name";
static constexpr const char* kProcessGroupDesc = "Process Group Description";
static constexpr const char* kGroupRanks = "Process Group Ranks";
static constexpr const char* kInTensorsStart = "Input Tensors start";
static constexpr const char* kOutTensorsStart = "Output Tensors start";
static constexpr const char* kRank = "Rank";
static constexpr const char* kP2pSrc = "Src Rank";
static constexpr const char* kP2pDst = "Dst Rank";

#ifdef __linux__
static constexpr char kDefaultLogFileFmt[] =
    "/tmp/libkineto_activities_{}.json";
#else
static constexpr char kDefaultLogFileFmt[] = "libkineto_activities_{}.json";
#endif

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}

// The 'ts' field written into the json file has 19 significant digits,
// while a double can only represent 15-16 digits. By using relative time,
// other applications can accurately read the 'ts' field as a double.
// Use the program loading time as the baseline time.
inline int64_t transToRelativeTime(int64_t time) {
  // Sometimes after converting to relative time, it can be a few nanoseconds
  // negative. Since Chrome trace and json processing will throw a parser error,
  // guard this.
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  if (res < 0) {
    return 0;
  }
  return res;
}

void ChromeTraceLogger::sanitizeStrForJSON(std::string& value) {
  // Replace all backslashes with forward slash because Windows paths causing
  // JSONDecodeError.
  std::replace(value.begin(), value.end(), '\\', '/');
  // Remove all new line characters
  value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
}

static std::string string2hex(const std::string& str) {
  std::string out;
  out.reserve(str.size() * 2);
  for (uint8_t c : str) {
    // “:02x” -> two‐digit, zero‐padded, lowercase hex
    fmt::format_to(std::back_inserter(out), "{:02x}", c);
  }
  return out;
}

static void sanitizeForNonReadableChars(std::string& value) {
  for (auto& c : value) {
    if (!std::isprint(c)) {
      LOG(WARNING) << "Non JSON compliant character found in string: 0x"
                   << string2hex(value) << " Replacing with 'unknown'";
      value = "unknown";
      break;
    }
  }
}

static inline int32_t sanitizeTid(int32_t tid) {
  // Convert all negative tids to its positive value. Create a specific case
  // for INT_MIN so it is obvious how it is being handled.
  if (tid == INT_MIN) {
    return 0;
  }
  return std::abs(tid);
}

void ChromeTraceLogger::metadataToJSON(
    const std::unordered_map<std::string, std::string>& metadata) {
  for (auto [k, v] : metadata) {
    std::string sanitizedValue = v;
    // There is a seperate mechanism for recording distributedInfo in on-demand
    // so add a guard to prevent "double counting" in auto-trace.
    if (k == "distributedInfo") {
      distInfo_.distInfo_present_ = true;
    }
    sanitizeStrForJSON(sanitizedValue);
    fmt::print(
        traceOf_,
        R"JSON(
      "{}": {},)JSON",
        k,
        sanitizedValue);
  }
}

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {
  if (!traceOf_) {
    return;
  }
  std::string display_unit = "ms";
#ifdef DISPLAY_TRACE_IN_NS
  display_unit = "ns";
#endif
  fmt::print(
      traceOf_,
      R"JSON(
  {{
    "schemaVersion": {},)JSON",
      kSchemaVersion);
  fmt::print(
      traceOf_,
      R"JSON(
    "deviceProperties": [{}
    ],)JSON",
      device_properties);

  metadataToJSON(metadata);
  fmt::print(
      traceOf_,
      R"JSON(
  "displayTimeUnit": "{}",
  "baseTimeNanoseconds": {},
  "traceEvents": [)JSON",
      display_unit,
      ChromeTraceBaseTime::singleton().get());
}

static std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

void ChromeTraceLogger::openTraceFile() {
  tempFileName_ = fileName_ + ".tmp";
  traceOf_.open(tempFileName_, std::ofstream::out | std::ofstream::trunc);
  if (!traceOf_) {
    PLOG(ERROR) << "Failed to open '" << fileName_ << "'";
  } else {
    LOG(INFO) << "Tracing to temporary file " << fileName_;
  }
}

void ChromeTraceLogger::finalizeMemoryTrace(const std::string&, const Config&) {
  LOG(INFO) << "finalizeMemoryTrace not implemented for ChromeTraceLogger";
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName) {
  fileName_ = traceFileName.empty() ? defaultFileName() : traceFileName;
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile();
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
  time = transToRelativeTime(time);
  fmt::print(traceOf_, R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_labels", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "labels": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000, info.id,
      info.name,
      time/1000, time%1000, info.id,
      info.label,
      time/1000, time%1000, info.id,
      info.sortIndex);
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
  time = transToRelativeTime(time);
  fmt::print(traceOf_, R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "thread_sort_index", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000, info.deviceId, sanitizeTid(info.id),
      info.name,
      time/1000, time%1000, info.deviceId, sanitizeTid(info.id),
      info.sortIndex);
  // clang-format on
}

void ChromeTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // TOOD: reserve pid = -1 for overhead but we need to rethink how to scale
  // this for other metadata
  // clang-format off
  time = transToRelativeTime(time);
  fmt::print(traceOf_, R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}.{:03}, "pid": -1, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03}, "pid": -1, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000,
      info.name,
      time/1000, time%1000,
      0x100000All);
  // clang-format on
}

void ChromeTraceLogger::handleTraceSpan(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  uint64_t start = transToRelativeTime(span.startTime);

  // If endTime is 0 and start time is non-zero, dur can overflow. Add
  // a guard to prevent this.
  uint64_t dur = (span.endTime == 0) ? 0 : span.endTime - span.startTime;
  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "X", "cat": "Trace", "ts": {}.{:03}, "dur": {}.{:03},
    "pid": "Spans", "tid": "{}",
    "name": "{}{} ({})",
    "args": {{
      "Op count": {}
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03},
    "pid": "Spans", "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      start/1000, start%1000, dur/1000, dur%1000,
      span.name,
      span.prefix, span.name, span.iteration,
      span.opCount,
      start/1000, start%1000,
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
  uint64_t start = transToRelativeTime(span.startTime);

  fmt::print(traceOf_, R"JSON(
  {{
    "name": "Iteration Start: {}", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace {}", "ts": {}.{:03}
  }},)JSON",
      span.name,
      span.name, start/1000, start%1000);
  // clang-format on
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  uint64_t ts = transToRelativeTime(op.timestamp());
  fmt::print(
      traceOf_,
      R"JSON(
  {{
    "ph": "i", "cat": "{}", "s": "t", "name": "{}",
    "pid": {}, "tid": {},
    "ts": {}.{:03},
    "args": {{
      {}
    }}
  }},)JSON",
      toString(op.type()),
      op.name(),
      op.deviceId(),
      sanitizeTid(op.resourceId()),
      ts / 1000,
      ts % 1000,
      op.metadataJson());
}

void ChromeTraceLogger::handleActivity(const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  int64_t ts = op.timestamp();
  int64_t duration = op.duration();

  if (duration < 0) {
    // This should never happen but can occasionally suffer from regression in
    // handling incomplete events. Having negative duration in Chrome trace can
    // yield in very poor experience so add an extra guard before we generate
    // trace events.
    duration = 0;
  }

  if (op.type() == ActivityType::GPU_USER_ANNOTATION) {
    // The GPU user annotations start at the same time as the
    // first associated GPU op. Since they appear later
    // in the trace file, this causes a visualization issue in Chrome.
    // Make it start one ns earlier and end 2 ns later.
    ts -= 1;
    duration += 2; // Still need it to end at the original point rounded up.
  }

  int external_id = 0;
  if (op.linkedActivity()) {
    external_id = op.linkedActivity()->correlationId();
  } else {
    // Some runtime events and kernels may not have a linked activity,
    // should not set an "External id" for them. Otherwise, these events
    // may be incorrectly linked to the other external events.
    static const std::set<libkineto::ActivityType> excludedTypes = {
        libkineto::ActivityType::GPU_MEMCPY,
        libkineto::ActivityType::GPU_MEMSET,
        libkineto::ActivityType::CONCURRENT_KERNEL,
        libkineto::ActivityType::CUDA_RUNTIME,
        libkineto::ActivityType::CUDA_DRIVER,
        libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
        libkineto::ActivityType::PRIVATEUSE1_DRIVER};
    if (excludedTypes.find(op.type()) == excludedTypes.end()) {
      external_id = op.correlationId();
    }
  }
  std::string arg_values;
  if (external_id != 0) {
    arg_values.append(fmt::format("\"External id\": {}", external_id));
  }
  std::string op_metadata = op.metadataJson();
  sanitizeStrForJSON(op_metadata);
  if (op_metadata.find_first_not_of(" \t\n") != std::string::npos) {
    if (!arg_values.empty()) {
      arg_values.append(",");
    }
    arg_values.append(op_metadata);
  }

  // Populate NCCL collective metadata from CPU to GPU
  if (op.type() == ActivityType::CONCURRENT_KERNEL && op.linkedActivity() &&
      op.linkedActivity()->name() == kParamCommsCallName) {
    const auto* collectiveRecord = op.linkedActivity();
    // Get the value out of the collective record
    const auto& collectiveName =
        collectiveRecord->getMetadataValue(kCollectiveName);
    const auto& inMsgSize = collectiveRecord->getMetadataValue(kInMsgNelems);
    const auto& outMsgSize = collectiveRecord->getMetadataValue(kOutMsgNelems);
    const auto& groupSize = collectiveRecord->getMetadataValue(kGroupSize);
    const auto& dtype = collectiveRecord->getMetadataValue(kDtype);
    if (!collectiveName.empty() && !inMsgSize.empty() && !outMsgSize.empty() &&
        !groupSize.empty() && !dtype.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(fmt::format(
          R"( "{}": {}, "{}": {}, "{}": {}, "{}": {}, "{}": {})",
          kCollectiveName,
          collectiveName,
          kInMsgNelems,
          inMsgSize,
          kOutMsgNelems,
          outMsgSize,
          kGroupSize,
          groupSize,
          kDtype,
          dtype));
    }
    const auto& input_tensor_starts =
        collectiveRecord->getMetadataValue(kInTensorsStart);
    const auto output_tensor_starts =
        collectiveRecord->getMetadataValue(kOutTensorsStart);
    if (!input_tensor_starts.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(
          fmt::format(" \"{}\": {}", kInTensorsStart, input_tensor_starts));
    }
    if (!output_tensor_starts.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(
          fmt::format(" \"{}\": {}", kOutTensorsStart, output_tensor_starts));
    }
    // In/out split size are valid for all_to_all
    const auto& inSplitSize = collectiveRecord->getMetadataValue(kInSplit);
    const auto& outSplitSize = collectiveRecord->getMetadataValue(kOutSplit);
    if (!inSplitSize.empty() && !outSplitSize.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(fmt::format(
          R"( "{}": {}, "{}": {})",
          kInSplit,
          inSplitSize,
          kOutSplit,
          outSplitSize));
    }
    const auto& processGroupName =
        collectiveRecord->getMetadataValue(kProcessGroupName);
    if (!processGroupName.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(
          fmt::format(" \"{}\": {}", kProcessGroupName, processGroupName));
    }
    const auto& processGroupDesc =
        collectiveRecord->getMetadataValue(kProcessGroupDesc);
    if (!processGroupName.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(
          fmt::format(" \"{}\": {}", kProcessGroupDesc, processGroupDesc));
    }
    const auto& groupRanks = collectiveRecord->getMetadataValue(kGroupRanks);
    if (!groupRanks.empty()) {
      if (!arg_values.empty()) {
        arg_values.append(",");
      }
      arg_values.append(fmt::format(" \"{}\": {}", kGroupRanks, groupRanks));
    }
    const auto& dstRank = collectiveRecord->getMetadataValue(kP2pDst);
    const auto& srcRank = collectiveRecord->getMetadataValue(kP2pSrc);
    if (!dstRank.empty()) {
      arg_values.append(fmt::format(", \"{}\": {}", kP2pDst, dstRank));
    }
    if (!srcRank.empty()) {
      arg_values.append(fmt::format(", \"{}\": {}", kP2pSrc, srcRank));
    }

    if (distInfo_.backend == "" && processGroupDesc == "\"default_pg\"") {
      distInfo_.backend = "nccl";
      distInfo_.rank = collectiveRecord->getMetadataValue(kRank);
      distInfo_.world_size = groupSize;
      // Not sure if we want to have output.json depend on nccl at compilation
      // so set nccl_version to "unknown" for now until we can determine if we
      // can pass it at runtime or use ifdefs. Should not be necessary to enable
      // HTA
      distInfo_.nccl_version = "unknown";
    }
    auto pg_config = pgConfig();
    pg_config.pg_name = processGroupName;
    pg_config.pg_desc = processGroupDesc;
    pg_config.backend_config = "cuda:nccl";
    pg_config.pg_size = groupSize;
    pg_config.ranks = groupRanks;
    pgMap.insert({processGroupName, pg_config});
  }

  std::string args;
  if (!arg_values.empty()) {
    args = fmt::format(
        R"JSON(,
    "args": {{
      {}
    }})JSON",
        arg_values);
  }

  int device = op.deviceId();
  int resource = op.resourceId();
  // TODO: Remove this once legacy tools are updated.
  std::string op_name = op.name() == "kernel" ? "Kernel" : op.name();
  sanitizeStrForJSON(op_name);
  sanitizeForNonReadableChars(op_name);

  // clang-format off
  ts = transToRelativeTime(ts);
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "X", "cat": "{}", "name": "{}", "pid": {}, "tid": {},
    "ts": {}.{:03}, "dur": {}.{:03}{}
  }},)JSON",
          toString(op.type()), op_name, device, sanitizeTid(resource),
          ts/1000, ts %1000, duration/1000, duration %1000, args);
  // clang-format on
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
  if (!traceOf_) {
    return;
  }
  handleActivity(op);
}

void ChromeTraceLogger::handleGenericLink(const ITraceActivity& act) {
  if (!traceOf_) {
    return;
  }
  static struct {
    int type;
    char name[16];
  } flow_names[] = {{kLinkFwdBwd, "fwdbwd"}, {kLinkAsyncCpuGpu, "ac2g"}};
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      // Link the activities via flow ID in source and destination.
      // The source node must return true from flowStart()
      // and the destination node false.
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.name);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.name);
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
    const std::string& name) {
  if (!traceOf_) {
    return;
  }

  // Flow events much bind to specific slices in order to exist.
  // Only Flow end needs to specify a binding point to enclosing slice.
  // Flow start automatically sets binding point to enclosing slice.
  const auto binding = (type == kFlowEnd) ? R"(, "bp": "e")" : "";
  // clang-format off
  uint64_t ts = transToRelativeTime(e.timestamp());
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "{}", "id": {}, "pid": {}, "tid": {}, "ts": {}.{:03},
    "cat": "{}", "name": "{}"{}
  }},)JSON",
      type, id, e.deviceId(), sanitizeTid(e.resourceId()), ts/1000, ts%1000, name, name, binding);
  // clang-format on
}

void ChromeTraceLogger::finalizeTrace(
    const Config& /*unused*/,
    std::unique_ptr<ActivityBuffers> /*unused*/,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  finalizeTrace(endTime, metadata);
}

void ChromeTraceLogger::addOnDemandDistMetadata() {
  if (distInfo_.backend == "") {
    return;
  }
  fmt::print(
      traceOf_,
      R"JSON(
  "distributedInfo": {{"backend": "{}", "rank": {}, "world_size": {}, "pg_count": {}, "pg_config": [)JSON",
      distInfo_.backend,
      distInfo_.rank,
      distInfo_.world_size,
      std::to_string(pgMap.size()));

  for (const auto& element : pgMap) {
    fmt::print(
        traceOf_,
        R"JSON({{"pg_name": {}, "pg_desc": {}, "backend_config": "{}", "pg_size": {}, "ranks": {}}},)JSON",
        element.second.pg_name,
        element.second.pg_desc,
        element.second.backend_config,
        element.second.pg_size,
        element.second.ranks);
  }
  traceOf_.seekp(-1, std::ios_base::end);
  fmt::print(
      traceOf_,
      R"JSON(], "nccl_version": "{}"}},)JSON",
      distInfo_.nccl_version);
  distInfo_.distInfo_present_ = true;
}

void ChromeTraceLogger::finalizeTrace(
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  sanitizeStrForJSON(fileName_);
  LOG(INFO) << "Chrome Trace written to " << fileName_;
  // clang-format off
  endTime = transToRelativeTime(endTime);
  fmt::print(traceOf_, R"JSON(
  {{
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": {}.{:03}
  }}
  ],)JSON",
      endTime/1000, endTime %1000);

  if (!distInfo_.distInfo_present_) {
   addOnDemandDistMetadata();
  }
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
      for (auto v : kv.second) {
        sanitizeStrForJSON(v);
        value.append("\"" + v + "\"");
        if(mdv_count > 1) {
          value.append(",");
          mdv_count--;
        }
      }
      value.append("]");
      PreparedMetadata[kv.first] = value;
    }
  }
  metadataToJSON(PreparedMetadata);
#endif // !USE_GOOGLE_LOG

  // Putting this here because the last entry MUST not end with a comma.
  fmt::print(traceOf_, R"JSON(
  "traceName": "{}"
}})JSON", fileName_);

  traceOf_.close();
  // On some systems, rename() fails if the destination file exists.
  // So, remove the destination file first.
  remove(fileName_.c_str());
  if (rename(tempFileName_.c_str(), fileName_.c_str()) != 0) {
    PLOG(ERROR) << "Failed to rename " << tempFileName_ << " to " << fileName_;
  } else {
    LOG(INFO) << "Renamed the trace file to " << fileName_;
  }
}

} // namespace KINETO_NAMESPACE
