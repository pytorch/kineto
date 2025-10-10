/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfiler.h"

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

inline bool XpuptiActivityProfilerSession::outOfScope(
    const ITraceActivity* act) {
  bool out_of_scope = act->timestamp() < captureWindowStartTime_ ||
      (act->timestamp() + act->duration()) > captureWindowEndTime_;
  if (out_of_scope) {
    std::string err_msg;
    err_msg += "TraceActivity outside of profiling window: " + act->name();
    err_msg += " (" + std::to_string(act->timestamp());
    err_msg += " < " + std::to_string(captureWindowStartTime_);
    err_msg += " or " + std::to_string(act->timestamp() + act->duration());
    err_msg += " > " + std::to_string(captureWindowEndTime_);
    errors_.push_back(err_msg);
  }
  return out_of_scope;
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

// FIXME: Deprecate this method while activity._sycl_queue_id got correct IDs
// from PTI
inline int64_t XpuptiActivityProfilerSession::getMappedQueueId(
    uint64_t sycl_queue_id) {
  auto [it, inserted] =
      sycl_queue_pool_.insert({sycl_queue_id, sycl_queue_pool_.size()});
  return it->second;
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
          "Invalid PTI External Correaltion activity sent to handlePtiActivity");
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

template <class pti_view_memory_record_type>
inline std::string bandwidth(pti_view_memory_record_type* activity) {
  auto duration = activity->_end_timestamp - activity->_start_timestamp;
  auto bytes = activity->_bytes;
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template <class pti_view_memory_record_type>
void XpuptiActivityProfilerSession::handleRuntimeKernelMemcpyMemsetActivities(
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
        traceBuffer_.span, ActivityType::XPU_RUNTIME, getApiName(activity));
  } else if constexpr (handleKernelActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        ActivityType::CONCURRENT_KERNEL,
        std::string(activity->_name));
  } else if constexpr (handleMemcpyActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        ActivityType::GPU_MEMCPY,
        memcpyName(
            activity->_memcpy_type, activity->_mem_src, activity->_mem_dst));
  } else if constexpr (handleMemsetActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        ActivityType::GPU_MEMSET,
        fmt::format(
            "Memset ({})", ptiViewMemoryTypeToString(activity->_mem_type)));
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
    trace_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
    trace_activity->flow.start = 0;
  }

  if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  }

  if constexpr (!handleRuntimeActivities) {
    trace_activity->addMetadata("appended", activity->_append_timestamp);
    trace_activity->addMetadata("submitted", activity->_submit_timestamp);
    trace_activity->addMetadata("device", trace_activity->deviceId());
    trace_activity->addMetadataQuoted(
        "context", handleToHexString(activity->_context_handle));
    trace_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
    trace_activity->addMetadataQuoted(
        "l0 queue", handleToHexString(activity->_queue_handle));
  }

  trace_activity->addMetadata("correlation", activity->_correlation_id);

  if constexpr (handleKernelActivities) {
    trace_activity->addMetadata("kernel_id", activity->_kernel_id);
  } else if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadata("memory opration id", activity->_mem_op_id);
    trace_activity->addMetadata("bytes", activity->_bytes);
    trace_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  }

  checkTimestampOrder(trace_activity.get());
  if (outOfScope(trace_activity.get())) {
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

  if (!outOfScope(overhead_activity.get())) {
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
          reinterpret_cast<const pti_view_record_api_t*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_KERNEL:
      handleRuntimeKernelMemcpyMemsetActivities(
          reinterpret_cast<const pti_view_record_kernel*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleRuntimeKernelMemcpyMemsetActivities(
          reinterpret_cast<const pti_view_record_memory_copy*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleRuntimeKernelMemcpyMemsetActivities(
          reinterpret_cast<const pti_view_record_memory_fill*>(record), logger);
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

#if PTI_VERSION_AT_LEAST(0, 14)

static std::string PtiValueToStr(
    pti_metric_value_type valueType,
    const pti_value_t& value) {
  switch (valueType) {
#define CASE(T, FIELD)            \
  case PTI_METRIC_VALUE_TYPE_##T: \
    return std::to_string(value.FIELD)

    CASE(UINT32, ui32);
    CASE(UINT64, ui64);
    CASE(FLOAT32, fp32);
    CASE(FLOAT64, fp64);

#undef CASE

    case PTI_METRIC_VALUE_TYPE_BOOL8:
      return (value.b8 ? "true" : "false");

    default:
      return "unknown";
  }
}

namespace {

struct DisplayInfoToStrIterator {
  DisplayInfoToStrIterator(const pti_metric_scope_display_info_t* displayInfo)
      : displayInfo_(displayInfo) {}

  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = std::string;
  using pointer = value_type*;
  using reference = value_type&;

  bool operator==(const DisplayInfoToStrIterator& rhs) const {
    return displayInfo_ == rhs.displayInfo_;
  }

  bool operator!=(const DisplayInfoToStrIterator& rhs) const {
    return !(*this == rhs);
  }

  std::string operator*() const {
    std::string s = std::string(displayInfo_->_name) + ": ";
    s += PtiValueToStr(displayInfo_->_value_type, displayInfo_->_value);

    if (displayInfo_->_units) {
      s += " ";
      s += displayInfo_->_units;
    }

    return s;
  }

  DisplayInfoToStrIterator& operator++() {
    ++displayInfo_;
    return *this;
  }

  const pti_metric_scope_display_info_t* displayInfo_ = nullptr;
};

struct DisplayInfoToStr {
  DisplayInfoToStr(const pti_metric_scope_display_info_t* v, size_t s)
      : begin_(v), end_(v + s) {}

  DisplayInfoToStrIterator begin() const {
    return DisplayInfoToStrIterator(begin_);
  }
  DisplayInfoToStrIterator end() const {
    return DisplayInfoToStrIterator(end_);
  }

  const pti_metric_scope_display_info_t* begin_ = nullptr;
  const pti_metric_scope_display_info_t* end_ = nullptr;
};

} // namespace

void XpuptiActivityProfilerSession::handleScopeRecord(
    const pti_metrics_scope_record_t* record,
    const pti_metric_scope_display_info_t* displayInfo,
    uint32_t infoCount,
    ActivityLogger& logger) {
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::XPU_SCOPE_PROFILER, "Scope");

  auto& scope_activity = traceBuffer_.activities.back();
  scope_activity->addMetadata("kernel id", record->_kernel_id);
  scope_activity->addMetadata("queue", record->_queue);
  scope_activity->addMetadata("name", record->_name);
  scope_activity->addMetadataQuoted(
      "metrics",
      fmt::format(
          "{}", fmt::join(DisplayInfoToStr(displayInfo, infoCount), ", ")));

  if (!outOfScope(scope_activity.get())) {
    scope_activity->log(logger);
  }
}

#endif

} // namespace KINETO_NAMESPACE
