/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"
#include "CuptiActivityBuffer.h"

#include <atomic>
#ifdef HAS_CUPTI
#include <cupti.h>
#include <cuda_occupancy.h>
#endif
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <vector>

namespace KINETO_NAMESPACE {

using namespace libkineto;

#ifndef HAS_CUPTI
using CUpti_Activity = void;
#endif

class CuptiActivityInterface {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  CuptiActivityInterface() = default;
  CuptiActivityInterface(const CuptiActivityInterface&) = delete;
  CuptiActivityInterface& operator=(const CuptiActivityInterface&) = delete;

  virtual ~CuptiActivityInterface() {}

  static CuptiActivityInterface& singleton();

  virtual int smCount();
  #ifdef HAS_CUPTI
  virtual std::vector<cudaOccDeviceProp> occDeviceProps();
  #endif
  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void disableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  void addActivityBuffer(uint8_t* buffer, size_t validSize);
  virtual std::unique_ptr<std::list<CuptiActivityBuffer>> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      std::list<CuptiActivityBuffer>& buffers,
      std::function<void(const CUpti_Activity*)> handler);

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

  int maxGpuBufferCount_{0};
  int allocatedGpuBufferCount{0};
  std::unique_ptr<std::list<CuptiActivityBuffer>> gpuTraceBuffers_;

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
