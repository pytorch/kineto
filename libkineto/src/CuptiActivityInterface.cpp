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
// Set to 1MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
constexpr size_t kBufSize(1 * 1024 * 1024);
constexpr size_t kAlignSize(64);

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

void CuptiActivityInterface::flushActivities() {
  CUPTI_CALL(cuptiActivityFlushAll(0));
}

const CUpti_Activity* CuptiActivityInterface::nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size) {
  CUptiResult status =
      cuptiActivityGetNextRecord(buffer, valid_size, &currentRecord_);
  if (status != CUPTI_SUCCESS) {
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      CUPTI_CALL(status);
    }
    currentRecord_ = nullptr;
  }
  return currentRecord_;
}

void CuptiActivityInterface::setMaxGpuBufferSize(int size) {
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
  void* bfr = malloc(kBufSize + kAlignSize);
  size_t sz = kBufSize + kAlignSize;
  *buffer = (uint8_t*)std::align(kAlignSize, kBufSize, bfr, sz);

  singleton().allocatedGpuBufferCount++;
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
