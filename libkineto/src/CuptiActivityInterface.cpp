/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivityInterface.h"

#include <chrono>

#include "Logger.h"
#include "cupti_call.h"
#include "external_api.h"

using namespace std::chrono;
using namespace libkineto;

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

void CuptiActivityInterface::pushCorrelationID(int id) {
  VLOG(2) << "pushCorrelationID(" << id << ")";
  CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id));
}

void CuptiActivityInterface::popCorrelationID() {
  CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr));
}

static int getSMCount() {
  // There may be a simpler way to get the number of SMs....
  // Look for domain_d - this has 80 instances on Volta and
  // 56 instances on Pascal, corresponding to the number of SMs
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

  return -1;
}

int CuptiActivityInterface::smCount() {
  static int sm_count = getSMCount();
  return sm_count;
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    CUpti_Activity*& record) {
  CUptiResult status =
      cuptiActivityGetNextRecord(buffer, valid_size, &record);
  if (status != CUPTI_SUCCESS) {
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      CUPTI_CALL(status);
    }
    record = nullptr;
  }
  return record != nullptr;
}

void CuptiActivityInterface::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}


void CUPTIAPI CuptiActivityInterface::bufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  if (singleton().allocatedGpuBufferCount > singleton().maxGpuBufferCount_) {
    // Stop profiling if we hit the max allowance
    external_api::setProfileRequestActive(false);
    singleton().stopCollection = true;
    LOG(WARNING) << "Exceeded max GPU buffer count ("
                 << singleton().allocatedGpuBufferCount
                 << ") - terminating tracing";
  }

  *size = kBufSize;
  *maxNumRecords = 0;

  // TODO(xdwang): create a list of buffers in advance so that we can reuse.
  // This saves time to dynamically allocate new buffers (which could be costly
  // if we allocated new space from the heap)
  *buffer = (uint8_t*) malloc(kBufSize);

  singleton().allocatedGpuBufferCount++;
}

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

const std::pair<int, int> CuptiActivityInterface::processActivities(
    std::function<void(const CUpti_Activity*)> handler) {
  VLOG(0) << "Flushing GPU activity buffers";
  time_point<high_resolution_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = high_resolution_clock::now();
  }
  CUPTI_CALL(cuptiActivityFlushAll(0));
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
  }

  int count = 0;
  int bytes = 0;
  while (!gpuTraceQueue.empty()) {
    VLOG(1) << "Processing GPU buffer";
    // No lock needed - only accessed from this thread
    auto gpu_trace = gpuTraceQueue.front();
    gpuTraceQueue.pop();
    bytes += gpu_trace.first;
    count += processActivitiesForBuffer(
        gpu_trace.second, gpu_trace.first, handler);
    free(gpu_trace.second);
  }

  return {count, bytes};
}

void CuptiActivityInterface::clearActivities() {
  CUPTI_CALL(cuptiActivityFlushAll(0));
  while (!gpuTraceQueue.empty()) {
    // FIXME: We might want to make sure we reuse
    // the same memory during warmup and tracing.
    free(gpuTraceQueue.front().second);
    gpuTraceQueue.pop();
  }
}

void CUPTIAPI CuptiActivityInterface::bufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  singleton().allocatedGpuBufferCount--;

  // lock should be uncessary here, because gpuTraceQueue is read/written by
  // profilerLoop only. CUPTI should handle the cuptiActivityFlushAll and
  // bufferCompleted, so that there is no concurrency issues
  singleton().gpuTraceQueue.push(std::make_pair(validSize, buffer));

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

void CuptiActivityInterface::enableCuptiActivities() {
  static bool registered = false;
  if (!registered) {
    CUPTI_CALL(
        cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  }
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void CuptiActivityInterface::disableCuptiActivities() {
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
}

} // namespace KINETO_NAMESPACE
