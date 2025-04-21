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

static bool
nextActivityRecord(uint8_t* buffer, size_t valid_size, Pti_Activity*& record) {
#ifdef HAS_XPUPTI
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
#endif
  return record != nullptr;
}

void XpuptiActivityApi::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

void XpuptiActivityApi::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XpuptiActivityApi::bufferRequested(uint8_t** buffer, size_t* size) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedGpuTraceBuffers_.size() >= (size_t)maxGpuBufferCount_) {
    stopCollection = true;
  }

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
    std::function<void(const Pti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Pti_Activity* record{nullptr};
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
    std::function<void(const Pti_Activity*)> handler) {
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

#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
static void enableSpecifcRuntimeAPIsTracing() {
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueUSMFill_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueUSMFill2D_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueUSMMemcpy_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueUSMMemcpy2D_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueKernelLaunch_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1,
      pti_api_group_id::PTI_API_GROUP_SYCL,
      urEnqueueKernelLaunchCustomExp_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1,
      pti_api_group_id::PTI_API_GROUP_SYCL,
      urEnqueueCooperativeKernelLaunchExp_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueMemBufferFill_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueMemBufferRead_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueMemBufferWrite_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urEnqueueMemBufferCopy_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urUSMHostAlloc_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urUSMSharedAlloc_id));
  XPUPTI_CALL(ptiViewEnableRuntimeApi(
      1, pti_api_group_id::PTI_API_GROUP_SYCL, urUSMDeviceAlloc_id));
}
#endif

void XpuptiActivityApi::enableXpuptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_XPUPTI
  XPUPTI_CALL(ptiViewSetCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::XPU_RUNTIME) {
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 11
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_RUNTIME_API));
      XPUPTI_CALL(ptiViewEnableRuntimeApiClass(
          1, PTI_API_CLASS_GPU_OPERATION_CORE, PTI_API_GROUP_ALL));
#elif PTI_VERSION_MAJOR == 0 && PTI_VERSION_MINOR == 11
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_RUNTIME_API));
      enableSpecifcRuntimeAPIsTracing();
#else
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_SYCL_RUNTIME_CALLS));
#endif
    }
    if (activity == ActivityType::OVERHEAD) {
      XPUPTI_CALL(ptiViewEnable(PTI_VIEW_COLLECTION_OVERHEAD));
    }
  }

  tracingEnabled_ = 1;
#endif

  stopCollection = false;
}

void XpuptiActivityApi::disablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_XPUPTI
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::XPU_RUNTIME) {
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_RUNTIME_API));
#else

      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_SYCL_RUNTIME_CALLS));
#endif
    }
    if (activity == ActivityType::OVERHEAD) {
      XPUPTI_CALL(ptiViewDisable(PTI_VIEW_COLLECTION_OVERHEAD));
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

} // namespace KINETO_NAMESPACE
