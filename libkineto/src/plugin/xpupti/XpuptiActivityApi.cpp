/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityApi.h"

#include <assert.h>
#include <chrono>
#include <mutex>
#include <thread>

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(4 * 1024 * 1024);

XpuptiActivityApi& XpuptiActivityApi::singleton() {
  static XpuptiActivityApi instance;
  return instance;
}

XpuptiActivityApi::XpuptiActivityApi() {
#ifdef HAS_XPUPTI
  XPUPTI_CALL(ptiMetricsGetDevices(nullptr, &device_count_));

  if (device_count_ > 0) {
    devices_ = std::make_unique<pti_device_properties_t[]>(device_count_);
    XPUPTI_CALL(ptiMetricsGetDevices(devices_.get(), &device_count_));
  }
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
  time_point<system_clock> t1;
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
  std::array<pti_api_id_runtime_sycl> specifcRuntimeAPIsTracing = {
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

void XpuptiActivityApi::enableScopeProfiler(const Config& cfg) {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (device_count_ == 0) {
    Error();
  }

  if (!scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeEnable(&*scopeHandleOpt_));

    const auto& spcfg = XpuptiScopeProfilerConfig::get(cfg);
    const auto& activitiesXpuptiMetrics = spcfg.activitiesXpuptiMetrics();

    activitiesXpuptiMetricsNames_.clean();
    activitiesXpuptiMetricsNames_.reserve(activitiesXpuptiMetrics.size());
    std::transform(
        activitiesXpuptiMetrics.begin(),
        activitiesXpuptiMetrics.end(),
        activitiesXpuptiMetricsNames_.begin(),
        [](const std::string& s) { return s.c_str(); });

    pti_metrics_scope_mode_t collection_mode = spcfg.xpuptiProfilerPerKernel()
        ? PTI_METRICS_SCOPE_AUTO_KERNEL
        : PTI_METRICS_SCOPE_USER;

    XPUPTI_CALL(ptiMetricsScopeConfigure(
        *scopeHandleOpt_,
        collection_mode,
        devices_,
        device_count_,
        activitiesXpuptiMetricsNames_.data(),
        activitiesXpuptiMetricsNames_.size()));
  }
#endif
#endif
}

void XpuptiActivityApi::disableScopeProfiler() {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeDisable(*scopeHandleOpt_));
    scopeHandleOpt_.reset();
  }
#endif
#endif
}

void XpuptiActivityApi::startScopedActivity() {
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

      case activity == ActivityType::XPU_RUNTIME:
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
        Error();
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

void XpuptiActivityApi::processScopeTrace(
    std::function<void(const pti_view_record_base*)> handler) {
#ifdef HAS_XPUPTI
#if PTI_VERSION_AT_LEAST(0, 14)
  if (scopeHandleOpt_) {
    uint32_t metricsCount = 0;
    XPUPTI_CALL(ptiMetricScopeGetCalculatedData(
        *scopeHandleOpt_, nullptr, &metricsCount));

    auto metricsValues = std::make_unique<pti_value_t[]>(metricsCount);
    XPUPTI_CALL(ptiMetricScopeGetCalculatedData(
        *scopeHandleOpt_, metricsValues.get(), &metricsCount));

    // How obtained pti_value_t results relate to the rest of the API calls ?

    uint64_t buffersCount = 0;
    XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffersCount(
        *scopeHandleOpt_, &buffersCount));

    for (uint64_t bufferId = 0; bufferId < buffersCount; ++bufferId) {
      size_t bufferSize = 0;
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffer(
          *scopeHandleOpt_, bufferId, nullptr, &bufferSize));

      XPUPTI_CALL(
          ptiMetricsScopeSetCollectionBuffer(*scopeHandleOpt_, bufferSize));

      void* buffer = nullptr;
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffer(
          *scopeHandleOpt_, bufferId, &buffer, &bufferSize));

      pti_metrics_scope_buffer_properties props;
      XPUPTI_CALL(
          ptiMetricsScopeGetBufferProperties(*scopeHandleOpt_, buffer, &props));

      size_t collectionBufferSize = 0;
      XPUPTI_CALL(ptiMetricsScopeQueryCollectionBufferSize(
          *scopeHandleOpt_, props._num_records, &collectionBufferSize));

      auto collectionBuffer = std::make_unique<uint8_t[]>(collectionBufferSize);

      size_t metricsCount = 0;
      XPUPTI_CALL(ptiMetricScopeGetCollectedMetrics(
          *scopeHandleOpt_, collectionBuffer.get(), &metricsCount));

      pti_metrics_scope_record* records = nullptr;
      size_t recordsCount = 0;
      size_t recordArrayBytesSize = 0;
      XPUPTI_CALL(ptiMetricScopeCalculateMetrics(
          *scopeHandleOpt_,
          collectionBuffer,
          activitiesXpuptiMetricsNames_.data(),
          activitiesXpuptiMetricsNames_.size(),
          &records,
          &recordsCount,
          &recordArrayBytesSize));

      for (size_t recordId = 0; recordId < recordsCount; ++recordId) {
        handler(records + recordId);
      }

      // Buffer cleanup functions
      XPUPTI_CALL(ptiCollectionBufferRelease(buffer));
    }
    // XPUPTI_CALL(ptiCollectionBufferCleanupAll(void);
  }
#endif
#endif
}

} // namespace KINETO_NAMESPACE
