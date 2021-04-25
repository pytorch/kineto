/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivityInterface.h"

#include <chrono>

#include "cupti_call.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

// TODO: do we want this to be configurable?
// Set to 2MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
// Consider putting this on huge pages?
constexpr size_t kBufSize(2 * 1024 * 1024);

CuptiActivityInterface& CuptiActivityInterface::singleton() {
  static CuptiActivityInterface instance;
  return instance;
}

void CuptiActivityInterface::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_CUPTI
  VLOG(2) << "pushCorrelationID(" << id << ")";
  switch(type) {
    case Default:
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id));
        break;
    case User:
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, id));
  }
#endif
}

void CuptiActivityInterface::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_CUPTI
  switch(type) {
    case Default:
      CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr));
        break;
    case User:
      CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, nullptr));
  }
#endif
}

static int getSMCount() {
#ifdef HAS_CUPTI
  // There may be a simpler way to get the number of SMs....
  // Look for domain_d - this has 80 instances on Volta and
  // 56 instances on Pascal, corresponding to the number of SMs
  // FIXME: This does not work on Turing and later
  uint32_t domainCount{0};
  CUPTI_CALL(cuptiDeviceGetNumEventDomains(0, &domainCount));
  std::vector<CUpti_EventDomainID> ids(domainCount);
  size_t sz = sizeof(CUpti_EventDomainID) * domainCount;
  CUPTI_CALL(cuptiDeviceEnumEventDomains(0, &sz, ids.data()));
  for (CUpti_EventDomainID id : ids) {
    char name[16];
    name[0] = '\0';
    sz = sizeof(name);
    CUPTI_CALL(cuptiEventDomainGetAttribute(
        id, CUPTI_EVENT_DOMAIN_ATTR_NAME, &sz, name));
    if (strncmp(name, "domain_d", sz) == 0) {
      uint32_t count{0};
      sz = sizeof(count);
      CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
          0, id, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &sz, &count));
      return count;
    }
  }
#endif

  return -1;
}

int CuptiActivityInterface::smCount() {
  static int sm_count = getSMCount();
  return sm_count;
}

#ifdef HAS_CUPTI
static std::vector<cudaOccDeviceProp> getOccDeviceProp() {
  std::vector<cudaOccDeviceProp> occProps;
  int device_count;
  cudaError_t error_id = cudaGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != cudaSuccess) {
    return std::vector<cudaOccDeviceProp>();
  }
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    error_id = cudaGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != cudaSuccess) {
      return std::vector<cudaOccDeviceProp>();
    }
    cudaOccDeviceProp occProp;
    occProp = prop;
    occProps.push_back(occProp);
  }
  return occProps;
}

std::vector<cudaOccDeviceProp> CuptiActivityInterface::occDeviceProps() {
  static std::vector<cudaOccDeviceProp> occProps = getOccDeviceProp();
  return occProps;
}
#endif

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    CUpti_Activity*& record) {
#ifdef HAS_CUPTI
  CUptiResult status = CUPTI_CALL_NOWARN(
      cuptiActivityGetNextRecord(buffer, valid_size, &record));
  if (status != CUPTI_SUCCESS) {
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      CUPTI_CALL(status);
    }
    record = nullptr;
  }
#endif
  return record != nullptr;
}

void CuptiActivityInterface::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

#ifdef HAS_CUPTI
void CUPTIAPI CuptiActivityInterface::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void CuptiActivityInterface::bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  if (allocatedGpuBufferCount >= maxGpuBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max GPU buffer count ("
                 << allocatedGpuBufferCount
                 << ") - terminating tracing";
  }

  *size = kBufSize;
  *maxNumRecords = 0;

  // TODO(xdwang): create a list of buffers in advance so that we can reuse.
  // This saves time to dynamically allocate new buffers (which could be costly
  // if we allocated new space from the heap)
  *buffer = (uint8_t*) malloc(kBufSize);

  allocatedGpuBufferCount++;
}
#endif

std::unique_ptr<std::list<CuptiActivityBuffer>> CuptiActivityInterface::activityBuffers() {
#ifdef HAS_CUPTI
  VLOG(1) << "Flushing GPU activity buffers";
  time_point<high_resolution_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = high_resolution_clock::now();
  }
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
  }
#endif
  return std::move(gpuTraceBuffers_);
}

#ifdef HAS_CUPTI
int CuptiActivityInterface::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const CUpti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    CUpti_Activity* record{nullptr};
    while ((nextActivityRecord(buf, validSize, record))) {
      handler(record);
      ++count;
    }
  }
  return count;
}
#endif

const std::pair<int, int> CuptiActivityInterface::processActivities(
    std::list<CuptiActivityBuffer>& buffers,
    std::function<void(const CUpti_Activity*)> handler) {
  std::pair<int, int> res{0, 0};
#ifdef HAS_CUPTI
  for (auto& buf : buffers) {
    // No lock needed - only accessed from this thread
    res.first += processActivitiesForBuffer(buf.data, buf.validSize, handler);
    res.second += buf.validSize;
  }
#endif
  return res;
}

void CuptiActivityInterface::clearActivities() {
  CUPTI_CALL(cuptiActivityFlushAll(0));
  // FIXME: We might want to make sure we reuse
  // the same memory during warmup and tracing.
  // Also, try to use the amount of memory required
  // for active tracing during warmup.
  if (gpuTraceBuffers_) {
    gpuTraceBuffers_->clear();
  }
}

void CuptiActivityInterface::addActivityBuffer(uint8_t* buffer, size_t validSize) {
  if (!gpuTraceBuffers_) {
    gpuTraceBuffers_ = std::make_unique<std::list<CuptiActivityBuffer>>();
  }
  gpuTraceBuffers_->emplace_back(buffer, validSize);
}

#ifdef HAS_CUPTI
void CUPTIAPI CuptiActivityInterface::bufferCompletedTrampoline(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  singleton().bufferCompleted(ctx, streamId, buffer, 0, validSize);
}

void CuptiActivityInterface::bufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  allocatedGpuBufferCount--;

  // lock should be uncessary here, because gpuTraceBuffers is read/written by
  // profilerLoop only. CUPTI should handle the cuptiActivityFlushAll and
  // bufferCompleted, so that there is no concurrency issues
  addActivityBuffer(buffer, validSize);

  // report any records dropped from the queue; to avoid unnecessary cupti
  // API calls, we make it report only in verbose mode (it doesn't happen
  // often in our testing anyways)
  if (VLOG_IS_ON(1)) {
    size_t dropped = 0;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
}
#endif

void CuptiActivityInterface::enableCuptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_CUPTI
  static bool registered = false;
  if (!registered) {
    CUPTI_CALL(
        cuptiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline));
  }

  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::CUDA_RUNTIME) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    }
  }
#endif

  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void CuptiActivityInterface::disableCuptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_CUPTI
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::CUDA_RUNTIME) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    }
  }
#endif
}

} // namespace KINETO_NAMESPACE
