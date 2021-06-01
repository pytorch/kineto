/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RoctracerActivityInterface.h"

#include <cstring>
#include <chrono>

#include "Logger.h"
#include "output_base.h"

static size_t BUFFERSIZE=1<<20;

using namespace std::chrono;

namespace KINETO_NAMESPACE {

// TODO: do we want this to be configurable?
// Set to 2MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
// Consider putting this on huge pages?
constexpr size_t kBufSize(2 * 1024 * 1024);

RoctracerActivityInterface& RoctracerActivityInterface::singleton() {
  static RoctracerActivityInterface instance;
  return instance;
}

RoctracerActivityInterface::RoctracerActivityInterface() {
  gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

void RoctracerActivityInterface::pushCorrelationID(int id, CorrelationFlowType type) {
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

void RoctracerActivityInterface::popCorrelationID(CorrelationFlowType type) {
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
  //hipDeviceProp_t 
  return -1;
}

int RoctracerActivityInterface::smCount() {
  static int sm_count = getSMCount();
  return sm_count;
}

void RoctracerActivityInterface::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

int RoctracerActivityInterface::processActivities(
    ActivityLogger& logger) {
#if 0
    printf("logger.handleGenericActivity(): %lu\n", activity_.size());
    //for (int i = 0; i < activity_.size(); ++i) {
    for (int i = 0; i < allocatedGpuBufferCount; ++i) {
        logger.handleGenericActivity(activity_[i]);
    }
#endif
  for (auto& buffer : *gpuTraceBuffers_) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data + buffer.validSize);
    GenericTraceActivity a;

    while (record < end_record) {
      if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->process_id;
        a.resource = record->thread_id;

        a.startTime = record->begin_ns / 1000;
        a.endTime = record->end_ns / 1000;
        a.correlation = 0;

        a.activityType = ActivityType::CUDA_RUNTIME;
        a.activityName = std::string(name);

        a.linked = NULL;
      }
      else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->device_id;
        a.resource = record->queue_id;

        a.startTime = record->begin_ns / 1000;
        a.endTime = record->end_ns / 1000;
        a.correlation = 0;

        a.activityType = ActivityType::CONCURRENT_KERNEL;
        a.activityName = std::string(name);

        a.linked = NULL;
      }
      logger.handleGenericActivity(a);
      roctracer_next_record(record, &record);     
    }
  }
}

void RoctracerActivityInterface::clearActivities() {
  if (gpuTraceBuffers_) {
    gpuTraceBuffers_->clear();
  }
}

void RoctracerActivityInterface::addActivityBuffer(uint8_t* buffer, size_t validSize) {
  if (!gpuTraceBuffers_) {
    gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
  }
  gpuTraceBuffers_->emplace_back(buffer, validSize);
}

void RoctracerActivityInterface::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &allocatedGpuBufferCount = singleton().allocatedGpuBufferCount;
  auto &gpuTraceBuffers = singleton().gpuTraceBuffers_;
  ++allocatedGpuBufferCount;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
  printf("hip callback: %d\n", allocatedGpuBufferCount);
}

void RoctracerActivityInterface::hip_activity_callback(const char* begin, const char* end, void* arg)
{
#if 0
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  auto &allocatedGpuBufferCount = singleton().allocatedGpuBufferCount;
  auto &activity = singleton().activity_;

  printf("hip callback: %d\n", allocatedGpuBufferCount);

  while (record < end_record) {
    if (allocatedGpuBufferCount < activity.size()) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        GenericTraceActivity &a = activity[allocatedGpuBufferCount];
        a.device = record->process_id;
        a.resource = record->thread_id;

        a.startTime = record->begin_ns / 1000;
        a.endTime = record->end_ns / 1000;
        a.correlation = 0;

        a.activityType = ActivityType::CPU_OP;
        a.activityType = ActivityType::CUDA_RUNTIME;
        a.activityName = std::string(name);

        a.linked = NULL;

        ++allocatedGpuBufferCount;
    }
    //const char *name = roctracer_op_string(record->domain, record->op, record->kind);
    //printf("t_api=%s, id=%lu, begin=%lu, end=%lu, pid=%d, tid=%lu\n", name, record->correlation_id, record->begin_ns, record->end_ns, record->process_id, record->thread_id);
    roctracer_next_record(record, &record);
  }
#endif
}

void RoctracerActivityInterface::hcc_activity_callback(const char* begin, const char* end, void* arg)
{
  printf("hcc callback\n");
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  while (record < end_record) {
    //const char *name = roctracer_op_string(record->domain, record->op, record->kind); 
    //printf("t_op=%s, id=%lu, begin=%lu, end=%lu, gpuId=%d, queueId=%lu\n", name, record->correlation_id, record->begin_ns, record->end_ns, record->device_id, record->queue_id);
    roctracer_next_record(record, &record);
  }
}

void RoctracerActivityInterface::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  static bool registered = false;
  if (!registered) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);  // Magic encantation

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

#if 1
    // Log hip
    roctracer_properties_t hip_cb_properties;
    memset(&hip_cb_properties, 0, sizeof(roctracer_properties_t));
    //hip_cb_properties.buffer_size = 0xf0000;
    hip_cb_properties.buffer_size = 0xf00;
    //hip_cb_properties.buffer_callback_fun = hip_activity_callback;
    hip_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hip_cb_properties, &hipPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HIP_API, hipPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hipPool_);    // FIXME - logging on 1 thread for now
#endif
#if 0
    // Log hcc
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x40000;
    hcc_cb_properties.buffer_callback_fun = hcc_activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);
#endif
  }

#endif
}

void RoctracerActivityInterface::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER

    //FIXME:  flush

#endif
}

} // namespace KINETO_NAMESPACE
