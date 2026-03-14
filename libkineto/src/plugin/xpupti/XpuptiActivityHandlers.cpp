/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSession.h"
#include "output_json.h"

#include <iterator>
#include <type_traits>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace KINETO_NAMESPACE {

// =========== Session Private Methods ============= //
void XpuptiActivityProfilerSession::removeCorrelatedPtiActivities(
    const ITraceActivity* act1) {
  correlatedPtiActivities_.erase(act1->correlationId());
}

void XpuptiActivityProfilerSession::checkTimestampOrder(
    const ITraceActivity* act1) {
  auto [it, inserted] =
      correlatedPtiActivities_.insert({act1->correlationId(), act1});
  if (inserted) {
    return;
  }

  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::XPU_RUNTIME) {
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    std::string err_msg;
    err_msg += "GPU op timestamp (" + std::to_string(act2->timestamp());
    err_msg += ") < runtime timestamp (" + std::to_string(act1->timestamp());
    err_msg += ") by " + std::to_string(act1->timestamp() - act2->timestamp());
    err_msg += "us Name: " + act2->name();
    err_msg += " Device: " + std::to_string(act2->deviceId());
    err_msg += " Queue: " + std::to_string(act2->resourceId());
    errors_.push_back(err_msg);
  }
}

inline bool XpuptiActivityProfilerSession::outOfRange(
    const ITraceActivity* act) {
  bool outOfRange = act->timestamp() < captureWindowStartTime_ ||
      (act->timestamp() + act->duration()) > captureWindowEndTime_;
  if (outOfRange) {
    std::string err_msg;
    err_msg += "TraceActivity outside of profiling window: " + act->name();
    err_msg += " (" + std::to_string(act->timestamp());
    err_msg += " < " + std::to_string(captureWindowStartTime_);
    err_msg += " or " + std::to_string(act->timestamp() + act->duration());
    err_msg += " > " + std::to_string(captureWindowEndTime_);
    errors_.push_back(err_msg);
  }
  return outOfRange;
}

const ITraceActivity* XpuptiActivityProfilerSession::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    return cpuActivity_(it->second);
  }
  return nullptr;
}

template <class ze_handle_type>
inline std::string handleToHexString(ze_handle_type handle) {
  return fmt::format("0x{:016x}", reinterpret_cast<uintptr_t>(handle));
}

inline void XpuptiActivityProfilerSession::handleCorrelationActivity(
    const pti_view_record_external_correlation* correlation) {
  switch (correlation->_external_kind) {
    case PTI_VIEW_EXTERNAL_KIND_CUSTOM_0:
      cpuCorrelationMap_[correlation->_correlation_id] =
          correlation->_external_id;
      break;
    case PTI_VIEW_EXTERNAL_KIND_CUSTOM_1:
      userCorrelationMap_[correlation->_correlation_id] =
          correlation->_external_id;
      break;
    default:
      errors_.push_back(
          "Invalid PTI External Correlation activity sent to handlePtiActivity");
  }
}

std::string XpuptiActivityProfilerSession::getApiName(
    const pti_view_record_api_t* activity) {
#if PTI_VERSION_AT_LEAST(0, 11)
  const char* api_name = nullptr;
  XPUPTI_CALL(
      ptiViewGetApiIdName(activity->_api_group, activity->_api_id, &api_name));
  return std::string(api_name);
#else
  return std::string(activity->_name);
#endif
}

inline std::string memcpyName(
    pti_view_memcpy_type kind,
    pti_view_memory_type src,
    pti_view_memory_type dst) {
  return fmt::format(
      "Memcpy {} ({} -> {})",
      ptiViewMemcpyTypeToString(kind),
      ptiViewMemoryTypeToString(src),
      ptiViewMemoryTypeToString(dst));
}

inline std::string memsetName(pti_view_memory_type type, uint64_t val) {
  return fmt::format("Memset ({} -> {})", val, ptiViewMemoryTypeToString(type));
}

template <class pti_view_memory_record_type>
inline std::string bandwidth(pti_view_memory_record_type* activity) {
  auto duration = activity->_end_timestamp - activity->_start_timestamp;
  auto bytes = activity->_bytes;
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

void XpuptiActivityProfilerSession::addResouceInfo(
    int32_t device_id,
    int32_t sycl_queue_id) {
  if (std::find_if(
          resourceInfo_.begin(),
          resourceInfo_.end(),
          [device_id, sycl_queue_id](std::pair<int32_t, int32_t> pair) {
            return (pair.first == device_id) && (pair.second == sycl_queue_id);
          }) == resourceInfo_.end()) {
    resourceInfo_.emplace_back(device_id, sycl_queue_id);
  }
}

template <class T>
inline std::string formatTimeLikeOutputJson(T time) {
  return fmt::format("{}.{:03}", time / 1000, abs(time) % 1000);
}

inline int64_t signedFromUnsignedDiff(uint64_t time, uint64_t time_ref) {
  if (time >= time_ref) {
    return static_cast<int64_t>(time - time_ref);
  } else {
    return -static_cast<int64_t>(time_ref - time);
  }
}

static void addTimestampMetadata(
    GenericTraceActivity* trace_activity,
    std::string&& label,
    uint64_t time,
    uint64_t time_ref) {
  trace_activity->addMetadataQuoted(
      label, formatTimeLikeOutputJson(transToRelativeTime(time)));
  label += "_rel_to_start";
  trace_activity->addMetadataQuoted(
      label, formatTimeLikeOutputJson(signedFromUnsignedDiff(time, time_ref)));
}

template <class pti_view_memory_record_type>
void XpuptiActivityProfilerSession::handleRuntimeKernelMemcpyMemsetActivities(
    ActivityType activityType,
    const pti_view_memory_record_type* activity,
    ActivityLogger& logger) {
  constexpr bool handleRuntimeActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_api_t>;
  constexpr bool handleKernelActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_kernel>;
  constexpr bool handleMemcpyActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_memory_copy>;
  constexpr bool handleMemsetActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_memory_fill>;

  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);

  if constexpr (handleRuntimeActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span, activityType, getApiName(activity));
  } else if constexpr (handleKernelActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span, activityType, std::string(activity->_name));
  } else if constexpr (handleMemcpyActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        activityType,
        memcpyName(
            activity->_memcpy_type, activity->_mem_src, activity->_mem_dst));
  } else if constexpr (handleMemsetActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        activityType,
        memsetName(activity->_mem_type, activity->_value_for_set));
  }

  auto& trace_activity = traceBuffer_.activities.back();

  trace_activity->startTime = activity->_start_timestamp;
  trace_activity->endTime = activity->_end_timestamp;
  trace_activity->id = activity->_correlation_id;
  trace_activity->threadId = activity->_thread_id;
  trace_activity->flow.id = activity->_correlation_id;
  trace_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  trace_activity->linked = linked;

  if constexpr (handleRuntimeActivities) {
    trace_activity->device = activity->_process_id;
    trace_activity->resource = activity->_thread_id;
    trace_activity->flow.start =
        (correlateRuntimeOps_.count(trace_activity->name()) > 0);
  } else {
    trace_activity->device = getDeviceIdxFromUUID(activity->_device_uuid);
    trace_activity->resource = activity->_sycl_queue_id;
    trace_activity->flow.start = 0;

    addResouceInfo(trace_activity->device, trace_activity->resource);
  }

  if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  }

  if constexpr (!handleRuntimeActivities) {
    addTimestampMetadata(
        trace_activity.get(),
        "appended",
        activity->_append_timestamp,
        activity->_start_timestamp);
    addTimestampMetadata(
        trace_activity.get(),
        "submitted",
        activity->_submit_timestamp,
        activity->_start_timestamp);
    if constexpr (handleKernelActivities) {
      addTimestampMetadata(
          trace_activity.get(),
          "sycl_task_begin",
          activity->_sycl_task_begin_timestamp,
          activity->_start_timestamp);
      addTimestampMetadata(
          trace_activity.get(),
          "sycl_enqk_begin",
          activity->_sycl_enqk_begin_timestamp,
          activity->_start_timestamp);
    }
    trace_activity->addMetadata("device", trace_activity->deviceId());
    trace_activity->addMetadataQuoted(
        "context", handleToHexString(activity->_context_handle));
    trace_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
    trace_activity->addMetadataQuoted(
        "l0 queue", handleToHexString(activity->_queue_handle));
  }

  trace_activity->addMetadata("correlation", activity->_correlation_id);

  if constexpr (handleKernelActivities) {
    if (activity->_source_file_name) {
      trace_activity->addMetadataQuoted(
          "source_file_name", activity->_source_file_name);
      trace_activity->addMetadata(
          "source_line_number", activity->_source_line_number);
    }
    trace_activity->addMetadata("kernel_id", activity->_kernel_id);
    trace_activity->addMetadata("sycl_node_id", activity->_sycl_node_id);
    trace_activity->addMetadata(
        "sycl_invocation_id", activity->_sycl_invocation_id);
  } else if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadata("memory opration id", activity->_mem_op_id);
    trace_activity->addMetadata("bytes", activity->_bytes);
    trace_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  }

  checkTimestampOrder(trace_activity.get());
  if (outOfRange(trace_activity.get())) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(trace_activity.get());
    traceBuffer_.activities.pop_back();
    return;
  }
  trace_activity->log(logger);
}

void XpuptiActivityProfilerSession::handleOverheadActivity(
    const pti_view_record_overhead* activity,
    ActivityLogger& logger) {
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::OVERHEAD,
      ptiViewOverheadKindToString(activity->_overhead_kind));
  auto& overhead_activity = traceBuffer_.activities.back();
  overhead_activity->startTime = activity->_overhead_start_timestamp_ns;
  overhead_activity->endTime = activity->_overhead_end_timestamp_ns;
  overhead_activity->device = -1;
  overhead_activity->resource = activity->_overhead_thread_id;
  overhead_activity->threadId = activity->_overhead_thread_id;
  overhead_activity->addMetadata(
      "overhead cost", activity->_overhead_duration_ns);
  overhead_activity->addMetadataQuoted(
      "overhead occupancy",
      fmt::format(
          "{}\%",
          activity->_overhead_duration_ns / overhead_activity->duration()));
  overhead_activity->addMetadata("overhead count", activity->_overhead_count);

  if (!outOfRange(overhead_activity.get())) {
    overhead_activity->log(logger);
  }
}

void XpuptiActivityProfilerSession::handlePtiActivity(
    const pti_view_record_base* record,
    ActivityLogger& logger) {
  switch (record->_view_kind) {
    case PTI_VIEW_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const pti_view_record_external_correlation*>(
              record));
      break;
#if PTI_VERSION_AT_LEAST(0, 11)
    case PTI_VIEW_RUNTIME_API:
#else
    case PTI_VIEW_SYCL_RUNTIME_CALLS:
#endif
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::XPU_RUNTIME,
          reinterpret_cast<const pti_view_record_api_t*>(record),
          logger);
      break;
    case PTI_VIEW_DRIVER_API:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::XPU_DRIVER,
          reinterpret_cast<const pti_view_record_api_t*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_KERNEL:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::CONCURRENT_KERNEL,
          reinterpret_cast<const pti_view_record_kernel*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::GPU_MEMCPY,
          reinterpret_cast<const pti_view_record_memory_copy*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::GPU_MEMSET,
          reinterpret_cast<const pti_view_record_memory_fill*>(record),
          logger);
      break;
    case PTI_VIEW_COLLECTION_OVERHEAD:
      handleOverheadActivity(
          reinterpret_cast<const pti_view_record_overhead*>(record), logger);
      break;
    default:
      errors_.push_back(
          "Unexpected activity type: " + std::to_string(record->_view_kind));
      break;
  }
}

} // namespace KINETO_NAMESPACE
