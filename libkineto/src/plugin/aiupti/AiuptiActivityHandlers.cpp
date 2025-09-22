#include "AiuptiActivityProfiler.h"

#include <aiupti_runtime_cbid.h>

#include <string>

namespace KINETO_NAMESPACE {

// =========== Session Private Methods ============= //
void AiuptiActivityProfilerSession::removeCorrelatedPtiActivities(
    const ITraceActivity* act1) {
  const auto key = act1->correlationId();
  const auto& it = correlatedPtiActivities_.find(key);
  if (it != correlatedPtiActivities_.end())
    correlatedPtiActivities_.erase(key);
  return;
}

void AiuptiActivityProfilerSession::checkTimestampOrder(
    const ITraceActivity* act1) {
  const auto& it = correlatedPtiActivities_.find(act1->correlationId());
  if (it == correlatedPtiActivities_.end()) {
    correlatedPtiActivities_.insert({act1->correlationId(), act1});
    return;
  }

  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::CUDA_RUNTIME)
    std::swap(act1, act2);
  if (act1->timestamp() > act2->timestamp()) {
    std::string err_msg;
    err_msg += "AIU op timestamp (" + std::to_string(act2->timestamp());
    err_msg += ") < runtime timestamp (" + std::to_string(act1->timestamp());
    err_msg += ") by " + std::to_string(act1->timestamp() - act2->timestamp());
    err_msg += "us Name: " + act2->name();
    err_msg += " Device: " + std::to_string(act2->deviceId());
    err_msg += " Queue: " + std::to_string(act2->resourceId());
    errors_.push_back(err_msg);
  }
}

inline bool AiuptiActivityProfilerSession::outOfRange(
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

const ITraceActivity* AiuptiActivityProfilerSession::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end())
    return cpuActivity_(it->second);
  return nullptr;
}

template <class ze_handle_type>
inline std::string handleToHexString(ze_handle_type handle) {
  return fmt::format("0x{:016x}", reinterpret_cast<uintptr_t>(handle));
}

inline std::string runtimeCbidName(AIUpti_runtime_api_trace_cbid cbid) {
  switch (cbid) {
    case AIUPTI_RUNTIME_TRACE_CBID_INVALID:
      return "aiuINVALID";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB:
      return "aiuLaunchControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT:
      return "aiuLaunchControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMI:
      return "aiuLaunchDMIControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO:
      return "aiuLaunchDMOControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CMPT_STREAM:
      return "aiuLaunchComputeStream";
    case AIUPTI_RUNTIME_TRACE_CBID_INIT_GRAPH:
      return "aiuInitGraph";
    case AIUPTI_RUNTIME_TRACE_CBID_MALLOC:
      return "aiuMalloc";
    case AIUPTI_RUNTIME_TRACE_CBID_RESIZE_TENSOR_ALLOCATION:
      return "aiuResizeTensorAllocation";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_SUPER_NODE:
      return "aiuLaunchSuperNode";
    case AIUPTI_RUNTIME_TRACE_CBID_SUPER_NODE_EXECUTE:
      return "aiuSuperNodeExecution";
    case AIUPTI_RUNTIME_TRACE_CBID_GRAPH_EXECUTE:
      return "aiuGraphExecution";
    case AIUPTI_RUNTIME_TRACE_CBID_NODE_COMPUTE:
      return "aiuNodeCompute";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_CONVERT:
      return "aiuDataConvert";
    case AIUPTI_RUNTIME_TRACE_CBID_INIT_SCHEDULER:
      return "aiuInitScheduler";
    case AIUPTI_RUNTIME_TRACE_CBID_CREATE_VIRTUAL_ADDRESSES:
      return "aiuCreateVirtualAddresses";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_SCHEDULE_COMPUTE:
      return "aiuLaunchScheduleCompute";
    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_WAIT:
      return "aiuScheduleWait";
    case AIUPTI_RUNTIME_TRACE_CBID_PREPARE_DMAS:
      return "aiuPrepareDMAs";
    case AIUPTI_RUNTIME_TRACE_CBID_PREPARE_SYNC_RDMA:
      return "aiuPrepareAndSyncRDMA";
    case AIUPTI_RUNTIME_TRACE_CBID_CLEAR_CACHE:
      return "aiuClearCache";
    case AIUPTI_RUNTIME_TRACE_CBID_PRELOAD_CACHE:
      return "aiuPreloadCache";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_COMPUTE_STREAM:
      return "aiuLaunchComputeStream";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_1:
      return "aiuRDMABarrier1";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_POST_KEYS:
      return "aiuPostRDMAKeys";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_2:
      return "aiuRDMABarrier2";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_FETCH_KEYS:
      return "aiuFetchRDMAKeys";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_UPDATE_CBS:
      return "aiuUpdateRDMACBs";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_3:
      return "aiuRDMABarrier3";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_DEADLOCK_CHECK:
      return "aiuCheckRDMADeadlock";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_DTOF:
      return "aiuFileTransferDtoF";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_MTOF:
      return "aiuFileTransferMtoF";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOD:
      return "aiuFileTransferFtoD";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOM:
      return "aiuFileTransferFtoM";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_DTOH:
      return "aiuDataTransferDtoH";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_HTOD:
      return "aiuDataTransferHtoD";
    case AIUPTI_RUNTIME_TRACE_CBID_CLOCK_CALIBRATION:
      return "aiuClockCalibration";
    case AIUPTI_RUNTIME_TRACE_CBID_COMPILE_GRAPH:
      return "aiuCompileGraph";
    default:
      break;
  }
  return "Unknown CBID " + std::to_string(cbid);
}

void AiuptiActivityProfilerSession::handleRuntimeActivity(
    const AIUpti_ActivityAPI* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  cpuCorrelationMap_[activity->correlation_id] = 0; // fake add correlation
  const ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  auto cbIDName =
      runtimeCbidName((AIUpti_runtime_api_trace_cbid)activity->cbid);
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::CUDA_RUNTIME, cbIDName);
  auto& runtime_activity = traceBuffer_.activities.back();
  runtime_activity->startTime = activity->start;
  runtime_activity->endTime = activity->end;
  runtime_activity->id = activity->correlation_id;
  runtime_activity->device = activity->process_id;
  runtime_activity->resource = systemThreadId();
  runtime_activity->threadId = threadId();
  // TODO: verify the flow logic
  runtime_activity->flow.id = activity->correlation_id;
  runtime_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  runtime_activity->flow.start = bool(
      std::find(
          correlateRuntimeOps_.begin(), correlateRuntimeOps_.end(), cbIDName) !=
      correlateRuntimeOps_.end());
  runtime_activity->linked = linked;
  runtime_activity->addMetadata("correlation", activity->correlation_id);

  switch ((AIUpti_runtime_api_trace_cbid)activity->cbid) {
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMI:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO:
      runtime_activity->addMetadata("num_cbs", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_DTOF:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_MTOF:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOD:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOM:
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_DTOH:
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_HTOD:
      runtime_activity->addMetadata("bytes", activity->data);
      break;

    default:
      break;
  }

  // checkTimestampOrder(&*runtime_activity);
  // if (outOfRange(*runtime_activity)) {
  //   traceBuffer_.span.opCount -= 1;
  //   traceBuffer_.gpuOpCount -= 1;
  //   removeCorrelatedPtiActivities(&*runtime_activity);
  //   traceBuffer_.activities.pop_back();
  //   return;
  // }
  runtime_activity->log(*logger);
}

// Finds the first number in the string after an underscore or hyphen
// and replaces it with "[N]". The replaced number is returned so that
// it can be preservered in the fn_idx metadata field
inline std::string extractInvocationNumber(std::string& str) {
  size_t start = 0;
  while (start < str.size() - 1) {
    size_t sep_idx = str.find_first_of("_-", start);
    // If no underscore or hyphen is found, return empty string
    if (sep_idx == std::string::npos || sep_idx + 1 >= str.size())
      return "";
    size_t end = str.find_first_not_of("0123456789", sep_idx + 1);
    // If all remaining characters are digits, replace them with "[N]"
    if (end == std::string::npos) {
      std::string num = str.substr(sep_idx + 1);
      str.replace(sep_idx + 1, std::string::npos, "[N]");
      return num;
    }
    // If the next character is not a digit, search for the next underscore or
    // hyphen
    if (end - sep_idx == 1) {
      start = sep_idx + 1;
      continue;
    }
    // Replace the found number following the underscore or hyphen with "[N]"
    std::string num = str.substr(sep_idx + 1, end - sep_idx - 1);
    str.replace(sep_idx + 1, end - sep_idx, "[N]");
    return num;
  }
  return "";
}

void AiuptiActivityProfilerSession::handleKernelActivity(
    const AIUpti_ActivityCompute* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  cpuCorrelationMap_[activity->correlation_id] = 0; // fake add correlation
  const ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  std::string name = activity->name;
  std::string num = extractInvocationNumber(name);
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::CONCURRENT_KERNEL, name);
  auto& kernel_activity = traceBuffer_.activities.back();
  kernel_activity->startTime = activity->start;
  kernel_activity->endTime = activity->end;
  kernel_activity->id = activity->correlation_id;
  kernel_activity->device = activity->device_id;
  kernel_activity->resource = activity->stream_id;
  kernel_activity->threadId = activity->stream_id;
  kernel_activity->flow.id = activity->correlation_id;
  kernel_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  kernel_activity->flow.start = 0;
  kernel_activity->linked = linked;
  kernel_activity->addMetadata("queued", activity->queued);
  kernel_activity->addMetadata("submitted", activity->submitted);
  kernel_activity->addMetadata("device", kernel_activity->deviceId());
  kernel_activity->addMetadata("stream", 1);
  kernel_activity->addMetadataQuoted(
      "context", std::to_string(activity->context_id));
  kernel_activity->addMetadata("correlation", activity->correlation_id);
  if (num != "")
    kernel_activity->addMetadata("fn_idx", num);

  recordStream(kernel_activity->device, kernel_activity->resource);

  // checkTimestampOrder(&*kernel_activity);
  // if (outOfRange(*kernel_activity)) {
  //   traceBuffer_.span.opCount -= 1;
  //   traceBuffer_.gpuOpCount -= 1;
  //   removeCorrelatedPtiActivities(&*kernel_activity);
  //   traceBuffer_.activities.pop_back();
  //   return;
  // }
  kernel_activity->log(*logger);
}

template <class memory_activity_type>
inline std::string bandwidth(memory_activity_type* activity) {
  auto duration = activity->end - activity->start;
  auto bytes = activity->bytes;
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

inline std::string memoryCopyOperationName(uint8_t kind) {
  switch (kind) {
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemcpy* activity) {
  return activity->copy_kind * 100;
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemory* activity) {
  return 400;
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemset* activity) {
  return 500;
}

template <class memory_activity_type>
uint32_t AiuptiActivityProfilerSession::getResourceId(
    memory_activity_type* activity) {
  uint32_t overlap_offset = 0;
  uint32_t base_resource_id = getBaseResourceId(activity);
  const auto it =
      activeThreadMap_.find({activity->device_id, base_resource_id});
  if (it != activeThreadMap_.end()) {
    std::vector<int64_t>& last_active_times = (*it).second;
    bool found_useable_thread = false;
    for (auto i = 0; i < last_active_times.size(); ++i) {
      if (last_active_times[i] <= activity->start) {
        found_useable_thread = true;
        overlap_offset = i;
        last_active_times[i] = activity->end;
        break;
      }
    }
    if (!found_useable_thread) {
      last_active_times.push_back(activity->end);
      overlap_offset = last_active_times.size() - 1;
    }
    activeThreadMap_[{activity->device_id, base_resource_id}][overlap_offset] =
        activity->end;
  } else {
    activeThreadMap_[{activity->device_id, base_resource_id}] = {activity->end};
  }
  return base_resource_id + overlap_offset;
}

template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemcpy>(AIUpti_ActivityMemcpy* activity);
template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemory>(AIUpti_ActivityMemory* activity);
template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemset>(AIUpti_ActivityMemset* activity);

void AiuptiActivityProfilerSession::handleMemcpyActivity(
    const AIUpti_ActivityMemcpy* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  cpuCorrelationMap_[activity->correlation_id] = 0; // fake add correlation
  const ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::GPU_MEMCPY,
      fmt::format("Memcpy ({})", memoryCopyOperationName(activity->copy_kind)));
  auto& memcpy_activity = traceBuffer_.activities.back();
  memcpy_activity->startTime = activity->start;
  memcpy_activity->endTime = activity->end;
  memcpy_activity->id = activity->correlation_id;
  memcpy_activity->device = activity->device_id;
  memcpy_activity->resource = getResourceId(activity);
  memcpy_activity->threadId = activity->stream_id;
  memcpy_activity->flow.id = activity->correlation_id;
  memcpy_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memcpy_activity->flow.start = 0;
  memcpy_activity->linked = linked;
  memcpy_activity->addMetadataQuoted(
      "call", memoryCopyOperationName(activity->copy_kind));
  memcpy_activity->addMetadata("device", memcpy_activity->deviceId());
  memcpy_activity->addMetadataQuoted(
      "context", std::to_string(activity->context_id));
  memcpy_activity->addMetadata("correlation", activity->correlation_id);
  memcpy_activity->addMetadata("memory operation id", activity->copy_kind);
  memcpy_activity->addMetadata("bytes", activity->bytes);
  memcpy_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));

  if (memcpy_activity->resource == getBaseResourceId(activity)) {
    recordMemoryStream(
        memcpy_activity->device,
        memcpy_activity->resource,
        fmt::format(
            "Memcpy ({}):", memoryCopyOperationName(activity->copy_kind)));
  } else {
    recordMemoryStream(memcpy_activity->device, memcpy_activity->resource, " ");
  }
  // TODO(mamaral): verify if we can enable this
  // checkTimestampOrder(&*memcpy_activity);
  // if (outOfRange(*memcpy_activity)) {
  //   traceBuffer_.span.opCount -= 1;
  //   traceBuffer_.gpuOpCount -= 1;
  //   removeCorrelatedPtiActivities(&*memcpy_activity);
  //   traceBuffer_.activities.pop_back();
  //   return;
  // }
  memcpy_activity->log(*logger);
}

inline std::string memoryOperationName(uint8_t kind) {
  switch (kind) {
    case (uint8_t)AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
      return "Allocation";
    case (uint8_t)AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
      return "Release";
    default:
      break;
  }
  return "<unknown>";
}

void AiuptiActivityProfilerSession::handleMemoryActivity(
    const AIUpti_ActivityMemory* activity,
    ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::CUDA_DRIVER,
      fmt::format(
          "Memory ({})", memoryOperationName(activity->memory_operation_type)));
  auto& mem_activity = traceBuffer_.activities.back();
  mem_activity->startTime = activity->start;
  mem_activity->endTime = activity->end;
  mem_activity->id = activity->correlation_id;
  mem_activity->device = activity->device_id;
  mem_activity->resource = getResourceId(activity);
  mem_activity->threadId = activity->stream_id;
  mem_activity->flow.id = activity->correlation_id;
  mem_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  mem_activity->flow.start = 0;
  mem_activity->linked = linked;
  mem_activity->addMetadataQuoted(
      "call", memoryOperationName(activity->memory_operation_type));
  mem_activity->addMetadata("device", mem_activity->deviceId());
  mem_activity->addMetadataQuoted(
      "context", std::to_string(activity->process_id));
  mem_activity->addMetadata("correlation", activity->correlation_id);
  mem_activity->addMetadata(
      "memory operation id", activity->memory_operation_type);
  mem_activity->addMetadata("bytes", activity->bytes);
  mem_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));

  if (mem_activity->resource == getBaseResourceId(activity)) {
    recordMemoryStream(
        mem_activity->device, mem_activity->resource, "Memory management:");
  } else {
    recordMemoryStream(mem_activity->device, mem_activity->resource, " ");
  }
  // checkTimestampOrder(&*mem_activity);
  // if (outOfRange(*mem_activity)) {
  //   traceBuffer_.span.opCount -= 1;
  //   traceBuffer_.gpuOpCount -= 1;
  //   removeCorrelatedPtiActivities(&*mem_activity);
  //   traceBuffer_.activities.pop_back();
  //   return;
  // }
  mem_activity->log(*logger);

  // Create event for AIU memory view
  traceBuffer_.span.opCount += 1;
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::CPU_INSTANT_EVENT, "[memory]");
  auto& memory_event = traceBuffer_.activities.back();

  memory_event->startTime = activity->start;

  // Following convention where all memory events are put on the
  // CPU thread. "Device Type" will denote CPU vs. AIU memory events
  // 0 (CPU), 1 (AIU)
  memory_event->device = systemThreadId();
  memory_event->resource = systemThreadId();

  int64_t bytes = static_cast<int64_t>(activity->bytes);
  if (activity->memory_operation_type ==
      AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
    bytes *= -1;
  }
  totalAllocatedBytes_ += bytes;
  memory_event->addMetadata("Total Reserved", 0);
  memory_event->addMetadata("Total Allocated", totalAllocatedBytes_);
  memory_event->addMetadata("Bytes", bytes);
  memory_event->addMetadata("Addr", activity->address);
  memory_event->addMetadata("Device Id", activity->device_id);
  memory_event->addMetadata("Device Type", 1);

  memory_event->log(*logger);
}

void AiuptiActivityProfilerSession::handleMemsetActivity(
    const AIUpti_ActivityMemset* activity,
    ActivityLogger* logger) {
  // do not track memset events because they are the same as memory allocation
  // events
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  // TODO(mamaral): implement the libaiupti to add external correlation ID
  cpuCorrelationMap_[activity->correlation_id] = 0; // fake add correlation
  const ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::GPU_MEMSET, "Memset (Device)");
  auto& memset_activity = traceBuffer_.activities.back();
  memset_activity->startTime = activity->start;
  memset_activity->endTime = activity->end;
  memset_activity->id = activity->correlation_id;
  memset_activity->device = activity->device_id;
  // TODO (mcalman): investigate why memset activities are being processed out
  // of order This prevents us from using getResourceId which handles overlap
  memset_activity->resource = getBaseResourceId(activity);
  memset_activity->threadId = activity->stream_id;
  memset_activity->flow.id = activity->correlation_id;
  memset_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memset_activity->flow.start = 0;
  memset_activity->linked = linked;
  memset_activity->addMetadataQuoted("call", "Memset");
  memset_activity->addMetadata("device", memset_activity->deviceId());
  memset_activity->addMetadataQuoted(
      "context", std::to_string(activity->context_id));
  memset_activity->addMetadata("correlation", activity->correlation_id);
  memset_activity->addMetadata("bytes", activity->bytes);
  memset_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));

  if (memset_activity->resource == getBaseResourceId(activity)) {
    recordMemoryStream(
        memset_activity->device, memset_activity->resource, "Memset (Device):");
  } else {
    recordMemoryStream(memset_activity->device, memset_activity->resource, " ");
  }

  // TODO(mamaral): verify if we can enable this
  // checkTimestampOrder(&*memset_activity);
  // if (outOfRange(*memset_activity)) {
  //   traceBuffer_.span.opCount -= 1;
  //   traceBuffer_.gpuOpCount -= 1;
  //   removeCorrelatedPtiActivities(&*memset_activity);
  //   traceBuffer_.activities.pop_back();
  //   return;
  // }
  memset_activity->log(*logger);
}

void AiuptiActivityProfilerSession::handlePtiActivity(
    const AIUpti_Activity* record,
    ActivityLogger* logger) {
  switch (record->kind) {
    case (uint8_t)AIUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const AIUpti_ActivityAPI*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_CMPT:
      handleKernelActivity(
          reinterpret_cast<const AIUpti_ActivityCompute*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMCPY:
      handleMemcpyActivity(
          reinterpret_cast<const AIUpti_ActivityMemcpy*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMSET:
      handleMemsetActivity(
          reinterpret_cast<const AIUpti_ActivityMemset*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMORY:
      handleMemoryActivity(
          reinterpret_cast<const AIUpti_ActivityMemory*>(record), logger);
      break;
    default:
      errors_.push_back(
          "Unexpected activity type: " + std::to_string(record->kind));
      break;
  }
}

} // namespace KINETO_NAMESPACE
