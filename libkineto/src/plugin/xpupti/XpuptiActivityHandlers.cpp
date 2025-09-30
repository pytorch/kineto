#include "XpuptiActivityProfiler.h"

#include <string>

namespace KINETO_NAMESPACE {

// =========== Session Private Methods ============= //
void XpuptiActivityProfilerSession::removeCorrelatedPtiActivities(
    const ITraceActivity* act1) {
  const auto key = act1->correlationId();
  const auto& it = correlatedPtiActivities_.find(key);
  if (it != correlatedPtiActivities_.end()) {
    correlatedPtiActivities_.erase(key);
  }
  return;
}

void XpuptiActivityProfilerSession::checkTimestampOrder(
    const ITraceActivity* act1) {
  const auto& it = correlatedPtiActivities_.find(act1->correlationId());
  if (it == correlatedPtiActivities_.end()) {
    correlatedPtiActivities_.insert({act1->correlationId(), act1});
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
    const ITraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    std::string err_msg;
    err_msg += "TraceActivity outside of profiling window: " + act.name();
    err_msg += " (" + std::to_string(act.timestamp());
    err_msg += " < " + std::to_string(captureWindowStartTime_);
    err_msg += " or " + std::to_string(act.timestamp() + act.duration());
    err_msg += " > " + std::to_string(captureWindowEndTime_);
    errors_.push_back(err_msg);
  }
  return out_of_range;
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
  auto it = std::find(
      sycl_queue_pool_.begin(), sycl_queue_pool_.end(), sycl_queue_id);
  if (it != sycl_queue_pool_.end()) {
    return std::distance(sycl_queue_pool_.begin(), it);
  }
  sycl_queue_pool_.push_back(sycl_queue_id);
  return sycl_queue_pool_.size() - 1;
}

inline void XpuptiActivityProfilerSession::handleCorrelationActivity(
    const pti_view_record_external_correlation* correlation) {
  if (correlation->_external_kind == PTI_VIEW_EXTERNAL_KIND_CUSTOM_0) {
    cpuCorrelationMap_[correlation->_correlation_id] =
        correlation->_external_id;
  } else if (correlation->_external_kind == PTI_VIEW_EXTERNAL_KIND_CUSTOM_1) {
    userCorrelationMap_[correlation->_correlation_id] =
        correlation->_external_id;
  } else {
    errors_.push_back(
        "Invalid PTI External Correaltion activity sent to handlePtiActivity");
  }
}

void XpuptiActivityProfilerSession::handleRuntimeActivity(
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
    const pti_view_record_api* activity,
#else
    const pti_view_record_sycl_runtime* activity,
#endif
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
  const char* api_name = nullptr;
  XPUPTI_CALL(
      ptiViewGetApiIdName(activity->_api_group, activity->_api_id, &api_name));
#endif
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::XPU_RUNTIME,
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
      std::string(api_name));
#else
      std::string(activity->_name));
#endif
  auto& runtime_activity = traceBuffer_.activities.back();
  runtime_activity->startTime = activity->_start_timestamp;
  runtime_activity->endTime = activity->_end_timestamp;
  runtime_activity->id = activity->_correlation_id;
  runtime_activity->device = activity->_process_id;
  runtime_activity->resource = activity->_thread_id;
  runtime_activity->threadId = activity->_thread_id;
  runtime_activity->flow.id = activity->_correlation_id;
  runtime_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  runtime_activity->flow.start = bool(
      std::find(
          correlateRuntimeOps_.begin(),
          correlateRuntimeOps_.end(),
          runtime_activity->name()) != correlateRuntimeOps_.end());
  runtime_activity->linked = linked;
  runtime_activity->addMetadata("correlation", activity->_correlation_id);

  checkTimestampOrder(&*runtime_activity);
  if (outOfRange(*runtime_activity)) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(&*runtime_activity);
    traceBuffer_.activities.pop_back();
    return;
  }
  runtime_activity->log(*logger);
}

void XpuptiActivityProfilerSession::handleKernelActivity(
    const pti_view_record_kernel* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::CONCURRENT_KERNEL,
      std::string(activity->_name));
  auto& kernel_activity = traceBuffer_.activities.back();
  kernel_activity->startTime = activity->_start_timestamp;
  kernel_activity->endTime = activity->_end_timestamp;
  kernel_activity->id = activity->_correlation_id;
  kernel_activity->device = getDeviceIdxFromUUID(activity->_device_uuid);
  kernel_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  kernel_activity->threadId = activity->_thread_id;
  kernel_activity->flow.id = activity->_correlation_id;
  kernel_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  kernel_activity->flow.start = 0;
  kernel_activity->linked = linked;
  kernel_activity->addMetadata("appended", activity->_append_timestamp);
  kernel_activity->addMetadata("submitted", activity->_submit_timestamp);
  kernel_activity->addMetadata("device", kernel_activity->deviceId());
  kernel_activity->addMetadataQuoted(
      "context", handleToHexString(activity->_context_handle));
  kernel_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  kernel_activity->addMetadataQuoted(
      "l0 queue", handleToHexString(activity->_queue_handle));
  kernel_activity->addMetadata("correlation", activity->_correlation_id);
  kernel_activity->addMetadata("kernel_id", activity->_kernel_id);

  checkTimestampOrder(&*kernel_activity);
  if (outOfRange(*kernel_activity)) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(&*kernel_activity);
    traceBuffer_.activities.pop_back();
    return;
  }
  kernel_activity->log(*logger);
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

void XpuptiActivityProfilerSession::handleMemcpyActivity(
    const pti_view_record_memory_copy* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::GPU_MEMCPY,
      memcpyName(
          activity->_memcpy_type, activity->_mem_src, activity->_mem_dst));
  auto& memcpy_activity = traceBuffer_.activities.back();
  memcpy_activity->startTime = activity->_start_timestamp;
  memcpy_activity->endTime = activity->_end_timestamp;
  memcpy_activity->id = activity->_correlation_id;
  memcpy_activity->device = getDeviceIdxFromUUID(activity->_device_uuid);
  memcpy_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  memcpy_activity->threadId = activity->_thread_id;
  memcpy_activity->flow.id = activity->_correlation_id;
  memcpy_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memcpy_activity->flow.start = 0;
  memcpy_activity->linked = linked;
  memcpy_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  memcpy_activity->addMetadata("appended", activity->_append_timestamp);
  memcpy_activity->addMetadata("submitted", activity->_submit_timestamp);
  memcpy_activity->addMetadata("device", memcpy_activity->deviceId());
  memcpy_activity->addMetadataQuoted(
      "context", handleToHexString(activity->_context_handle));
  memcpy_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  memcpy_activity->addMetadataQuoted(
      "l0 queue", handleToHexString(activity->_queue_handle));
  memcpy_activity->addMetadata("correlation", activity->_correlation_id);
  memcpy_activity->addMetadata("memory opration id", activity->_mem_op_id);
  memcpy_activity->addMetadata("bytes", activity->_bytes);
  memcpy_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));

  checkTimestampOrder(&*memcpy_activity);
  if (outOfRange(*memcpy_activity)) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(&*memcpy_activity);
    traceBuffer_.activities.pop_back();
    return;
  }
  memcpy_activity->log(*logger);
}

void XpuptiActivityProfilerSession::handleMemsetActivity(
    const pti_view_record_memory_fill* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::GPU_MEMSET,
      fmt::format(
          "Memset ({})", ptiViewMemoryTypeToString(activity->_mem_type)));
  auto& memset_activity = traceBuffer_.activities.back();
  memset_activity->startTime = activity->_start_timestamp;
  memset_activity->endTime = activity->_end_timestamp;
  memset_activity->id = activity->_correlation_id;
  memset_activity->device = getDeviceIdxFromUUID(activity->_device_uuid);
  memset_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  memset_activity->threadId = activity->_thread_id;
  memset_activity->flow.id = activity->_correlation_id;
  memset_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memset_activity->flow.start = 0;
  memset_activity->linked = linked;
  memset_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  memset_activity->addMetadata("appended", activity->_append_timestamp);
  memset_activity->addMetadata("submitted", activity->_submit_timestamp);
  memset_activity->addMetadata("device", memset_activity->deviceId());
  memset_activity->addMetadataQuoted(
      "context", handleToHexString(activity->_context_handle));
  memset_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  memset_activity->addMetadataQuoted(
      "l0 queue", handleToHexString(activity->_queue_handle));
  memset_activity->addMetadata("correlation", activity->_correlation_id);
  memset_activity->addMetadata("memory opration id", activity->_mem_op_id);
  memset_activity->addMetadata("bytes", activity->_bytes);
  memset_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));

  checkTimestampOrder(&*memset_activity);
  if (outOfRange(*memset_activity)) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(&*memset_activity);
    traceBuffer_.activities.pop_back();
    return;
  }
  memset_activity->log(*logger);
}

void XpuptiActivityProfilerSession::handleOverheadActivity(
    const pti_view_record_overhead* activity,
    ActivityLogger* logger) {
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

  if (outOfRange(*overhead_activity)) {
    return;
  }
  overhead_activity->log(*logger);
}

void XpuptiActivityProfilerSession::handlePtiActivity(
    const pti_view_record_base* record,
    ActivityLogger* logger) {
  switch (record->_view_kind) {
    case PTI_VIEW_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const pti_view_record_external_correlation*>(
              record));
      break;
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
    case PTI_VIEW_RUNTIME_API:
      handleRuntimeActivity(
          reinterpret_cast<const pti_view_record_api*>(record),
#else
    case PTI_VIEW_SYCL_RUNTIME_CALLS:
      handleRuntimeActivity(
          reinterpret_cast<const pti_view_record_sycl_runtime*>(record),
#endif
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_KERNEL:
      handleKernelActivity(
          reinterpret_cast<const pti_view_record_kernel*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleMemcpyActivity(
          reinterpret_cast<const pti_view_record_memory_copy*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleMemsetActivity(
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

} // namespace KINETO_NAMESPACE
