/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityApi.h"
#include "XpuptiScopeProfilerConfig.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(4 * 1024 * 1024);

XpuptiActivityApi& XpuptiActivityApi::singleton() {
  static XpuptiActivityApi instance;
  return instance;
}

XpuptiActivityApi::XpuptiActivityApi() {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  XPUPTI_CALL(ptiMetricsGetDevices(nullptr, &deviceCount_));

  if (deviceCount_ > 0) {
    auto devices = std::make_unique<pti_device_properties_t[]>(deviceCount_);
    XPUPTI_CALL(ptiMetricsGetDevices(devices.get(), &deviceCount_));

    devicesHandles_ = std::make_unique<pti_device_handle_t[]>(deviceCount_);
    for (uint32_t i = 0; i < deviceCount_; ++i) {
      devicesHandles_[i] = devices[i]._handle;
    }
  }
#endif
#endif
}

void XpuptiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_XPUPTI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      XPUPTI_CALL(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, id));
      break;
    case User:
      XPUPTI_CALL(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, id));
  }
#endif
}

void XpuptiActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_XPUPTI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      XPUPTI_CALL(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, nullptr));
      break;
    case User:
      XPUPTI_CALL(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, nullptr));
  }
#endif
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    pti_view_record_base*& record) {
#ifdef HAS_XPUPTI
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
#endif
  return record != nullptr;
}

void XpuptiActivityApi::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XpuptiActivityApi::bufferRequested(uint8_t** buffer, size_t* size) {
  std::lock_guard<std::mutex> guard(mutex_);

  auto buf = std::make_unique<XpuptiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGpuTraceBuffers_[*buffer] = std::move(buf);
}

std::unique_ptr<XpuptiActivityBufferMap> XpuptiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return nullptr;
    }
  }

#ifdef HAS_XPUPTI
  std::chrono::time_point<std::chrono::system_clock> t1;
  XPUPTI_CALL(ptiFlushAllViews());
#endif

  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

#ifdef HAS_XPUPTI
int XpuptiActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const pti_view_record_base*)> handler) {
  int count = 0;
  if (buf && validSize) {
    pti_view_record_base* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}
#endif

const std::pair<int, int> XpuptiActivityApi::processActivities(
    XpuptiActivityBufferMap& buffers,
    std::function<void(const pti_view_record_base*)> handler) {
  std::pair<int, int> res{0, 0};
#ifdef HAS_XPUPTI
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
#endif
  return res;
}

void XpuptiActivityApi::flushActivities() {
#ifdef HAS_XPUPTI
  XPUPTI_CALL(ptiFlushAllViews());
#endif
}

void XpuptiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
#ifdef HAS_XPUPTI
  XPUPTI_CALL(ptiFlushAllViews());
#endif
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_ = nullptr;
}

#ifdef HAS_XPUPTI
void XpuptiActivityApi::bufferCompletedTrampoline(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void XpuptiActivityApi::bufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedGpuTraceBuffers_.find(buffer);

  if (!readyGpuTraceBuffers_) {
    readyGpuTraceBuffers_ = std::make_unique<XpuptiActivityBufferMap>();
  }
  it->second->setSize(validSize);
  (*readyGpuTraceBuffers_)[it->first] = std::move(it->second);
  allocatedGpuTraceBuffers_.erase(it);
}
#endif

#if PTI_VERSION_AT_LEAST(0, 11)
static void enableSpecifcRuntimeAPIsTracing() {
  constexpr const std::array<pti_api_id_runtime_sycl, 14>
      specifcRuntimeAPIsTracing = {
          urEnqueueUSMFill_id,
          urEnqueueUSMFill2D_id,
          urEnqueueUSMMemcpy_id,
          urEnqueueUSMMemcpy2D_id,
          urEnqueueKernelLaunch_id,
          urEnqueueKernelLaunchCustomExp_id,
          urEnqueueCooperativeKernelLaunchExp_id,
          urEnqueueMemBufferFill_id,
          urEnqueueMemBufferRead_id,
          urEnqueueMemBufferWrite_id,
          urEnqueueMemBufferCopy_id,
          urUSMHostAlloc_id,
          urUSMSharedAlloc_id,
          urUSMDeviceAlloc_id};

  for (auto tracing_id : specifcRuntimeAPIsTracing) {
    XPUPTI_CALL(ptiViewEnableRuntimeApi(
        1, pti_api_group_id::PTI_API_GROUP_SYCL, tracing_id));
  }
}
#endif

#if PTI_VERSION_AT_LEAST(0, 14)
XpuptiActivityApi::safe_pti_scope_collection_handle_t::
    safe_pti_scope_collection_handle_t() {
  XPUPTI_CALL(ptiMetricsScopeEnable(&handle));
}

XpuptiActivityApi::safe_pti_scope_collection_handle_t::
    ~safe_pti_scope_collection_handle_t() {
  XPUPTI_CALL(ptiMetricsScopeDisable(handle));
}
#endif

void XpuptiActivityApi::enableScopeProfiler(const Config& cfg) {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (deviceCount_ == 0) {
    throw std::runtime_error("No XPU devices available");
  }

  scopeHandleOpt_.emplace();

  const auto& spcfg = XpuptiScopeProfilerConfig::get(cfg);
  const auto& activitiesXpuptiMetrics = spcfg.activitiesXpuptiMetrics();

  std::vector<const char*> metricNames;
  metricNames.reserve(activitiesXpuptiMetrics.size());
  std::transform(
      activitiesXpuptiMetrics.begin(),
      activitiesXpuptiMetrics.end(),
      std::back_inserter(metricNames),
      [](const std::string& s) { return s.c_str(); });

  pti_metrics_scope_mode_t collectionMode = spcfg.xpuptiProfilerPerKernel()
      ? PTI_METRICS_SCOPE_AUTO_KERNEL
      : PTI_METRICS_SCOPE_USER;

  if (collectionMode == PTI_METRICS_SCOPE_USER) {
    throw std::runtime_error(
        "XPUPTI_PROFILER_ENABLE_PER_KERNEL has to be set to 1. Other variants are currently not supported.");
  }

  XPUPTI_CALL(ptiMetricsScopeConfigure(
      *scopeHandleOpt_,
      collectionMode,
      devicesHandles_.get(),
      (deviceCount_, 1), // Only 1 device is currently supported
      metricNames.data(),
      metricNames.size()));

  uint64_t expectedKernels = spcfg.xpuptiProfilerMaxScopes();
  size_t estimatedCollectionBufferSize = 0;
  XPUPTI_CALL(ptiMetricsScopeQueryCollectionBufferSize(
      *scopeHandleOpt_, expectedKernels, &estimatedCollectionBufferSize));

  XPUPTI_CALL(ptiMetricsScopeSetCollectionBufferSize(
      *scopeHandleOpt_, estimatedCollectionBufferSize));
#endif
#endif
}

void XpuptiActivityApi::disableScopeProfiler() {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  scopeHandleOpt_.reset();
#endif
#endif
}

void XpuptiActivityApi::startScopeActivity() {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeStartCollection(*scopeHandleOpt_));
  }
#endif
#endif
}

void XpuptiActivityApi::stopScopeActivity() {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeStopCollection(*scopeHandleOpt_));
  }
#endif
#endif
}

bool XpuptiActivityApi::enableXpuptiActivities(
    const std::set<ActivityType>& selected_activities) {
  bool scopeProfilerEnabled = false;
#ifdef HAS_XPUPTI
  XPUPTI_CALL(ptiViewSetCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    switch (activity) {
      case ActivityType::GPU_MEMCPY:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
        break;

      case ActivityType::GPU_MEMSET:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
        break;

      case ActivityType::CONCURRENT_KERNEL:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL));
        break;

      case ActivityType::EXTERNAL_CORRELATION:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION));
        externalCorrelationEnabled_ = true;
        break;

      case ActivityType::XPU_RUNTIME:
#if PTI_VERSION_AT_LEAST(0, 12)
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_RUNTIME_API));
        XPUPTI_CALL(ptiViewEnableRuntimeApiClass(
            1, PTI_API_CLASS_GPU_OPERATION_CORE, PTI_API_GROUP_ALL));
#elif PTI_VERSION_AT_LEAST(0, 11)
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_RUNTIME_API));
        enableSpecifcRuntimeAPIsTracing();
#else
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_SYCL_RUNTIME_CALLS));
#endif
        break;

      case ActivityType::XPU_SCOPE_PROFILER:
#if PTI_VERSION_AT_LEAST(0, 14)
        scopeProfilerEnabled = true;
#else
        throw std::runtime_error(
            "Intel® oneAPI version required to use scope profiler is at least 2025.3.0");
#endif
        break;

      case ActivityType::OVERHEAD:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_COLLECTION_OVERHEAD));
        break;
    }
  }
#endif
  return scopeProfilerEnabled;
}

void XpuptiActivityApi::disablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_XPUPTI
  for (const auto& activity : selected_activities) {
    switch (activity) {
      case ActivityType::GPU_MEMCPY:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
        break;

      case ActivityType::GPU_MEMSET:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
        break;

      case ActivityType::CONCURRENT_KERNEL:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL));
        break;

      case ActivityType::EXTERNAL_CORRELATION:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION));
        break;

      case ActivityType::XPU_RUNTIME:
#if PTI_VERSION_AT_LEAST(0, 11)
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_RUNTIME_API));
#else
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_SYCL_RUNTIME_CALLS));
#endif
        break;

      case ActivityType::XPU_SCOPE_PROFILER:
        // This case is handled by XpuptiActivityApi::disableScopeProfiler
        break;

      case ActivityType::OVERHEAD:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_COLLECTION_OVERHEAD));
        break;
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

#if PTI_VERSION_AT_LEAST(0, 14)

static size_t IntDivRoundUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

void XpuptiActivityApi::processScopeTrace(
    std::function<void(
        const pti_metrics_scope_record_t*,
        const pti_metrics_scope_record_metadata_t& metadata)> handler) {
#ifdef HAS_XPUPTI
  if (scopeHandleOpt_) {
    pti_metrics_scope_record_metadata_t metadata;
    metadata._struct_size = sizeof(pti_metrics_scope_record_metadata_t);

    XPUPTI_CALL(ptiMetricsScopeGetMetricsMetadata(*scopeHandleOpt_, &metadata));

    uint64_t collectionBuffersCount = 0;
    XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffersCount(
        *scopeHandleOpt_, &collectionBuffersCount));

    for (uint64_t bufferId = 0; bufferId < collectionBuffersCount; ++bufferId) {
      void* collectionBuffer = nullptr;
      size_t actualCollectionBufferSize = 0;
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffer(
          *scopeHandleOpt_,
          bufferId,
          &collectionBuffer,
          &actualCollectionBufferSize));

      pti_metrics_scope_collection_buffer_properties_t metricsBufferProps;
      metricsBufferProps._struct_size =
          sizeof(pti_metrics_scope_collection_buffer_properties_t);
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBufferProperties(
          *scopeHandleOpt_, collectionBuffer, &metricsBufferProps));

      size_t requiredMetricsBufferSize = 0;
      size_t recordsCount = 0;
      XPUPTI_CALL(ptiMetricsScopeQueryMetricsBufferSize(
          *scopeHandleOpt_,
          collectionBuffer,
          &requiredMetricsBufferSize,
          &recordsCount));

      if (recordsCount > 0) {
        auto metricsBuffer =
            std::make_unique<pti_metrics_scope_record_t[]>(IntDivRoundUp(
                requiredMetricsBufferSize, sizeof(pti_metrics_scope_record_t)));

        size_t actualRecordsCount = 0;
        XPUPTI_CALL(ptiMetricsScopeCalculateMetrics(
            *scopeHandleOpt_,
            collectionBuffer,
            metricsBuffer.get(),
            requiredMetricsBufferSize,
            &actualRecordsCount));

        for (size_t recordId = 0; recordId < actualRecordsCount; ++recordId) {
          auto record = metricsBuffer.get() + recordId;
          handler(record, metadata);
        }
      }
    }
  }
#endif
}
#endif

} // namespace KINETO_NAMESPACE
