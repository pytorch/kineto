/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"
#include "RoctracerActivityBuffer.h"	// FIXME remove me
#include "GenericTraceActivity.h"

#include <atomic>
#ifdef HAS_ROCTRACER
#include <roctracer.h>
#endif
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <vector>

namespace KINETO_NAMESPACE {

using namespace libkineto;


class RoctracerActivityInterface {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  RoctracerActivityInterface();
  RoctracerActivityInterface(const RoctracerActivityInterface&) = delete;
  RoctracerActivityInterface& operator=(const RoctracerActivityInterface&) = delete;

  virtual ~RoctracerActivityInterface() {}

  static RoctracerActivityInterface& singleton();

  virtual int smCount();
  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(
    const std::set<ActivityType>& selected_activities);
  void disableActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  void addActivityBuffer(uint8_t* buffer, size_t validSize);
  //virtual std::unique_ptr<std::list<RoctracerActivityBuffer>> activityBuffers();

  //virtual const std::pair<int, int> processActivities(
  //    std::list<CuptiActivityBuffer>& buffers,
  //    std::function<void(const CUpti_Activity*)> handler);

  int processActivities(ActivityLogger& logger);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
#ifdef HAS_CUPTI
  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const CUpti_Activity*)> handler);
  static void CUPTIAPI
  bufferRequestedTrampoline(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  static void CUPTIAPI bufferCompletedTrampoline(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);
#endif // HAS_CUPTI

#ifdef HAS_ROCTRACER
  roctracer_pool_t *hipPool_{NULL};
  roctracer_pool_t *hccPool_{NULL};
  static void activity_callback(const char* begin, const char* end, void* arg);
  static void hip_activity_callback(const char* begin, const char* end, void* arg);
  static void hcc_activity_callback(const char* begin, const char* end, void* arg);
#endif

  int maxGpuBufferCount_{0};
  int allocatedGpuBufferCount{0};
  std::unique_ptr<std::list<RoctracerActivityBuffer>> gpuTraceBuffers_;
  //int eventCount{0};
  //std::vector<GenericTraceActivity> hipActivity_;
  //std::vector<GenericTraceActivity> hccActivity_;


 protected:
#ifdef HAS_CUPTI
  void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);
#endif
};

} // namespace KINETO_NAMESPACE

