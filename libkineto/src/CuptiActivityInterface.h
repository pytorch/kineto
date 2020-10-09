/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "CuptiActivityType.h"

#include <atomic>
#include <cupti.h>
#include <functional>
#include <queue>
#include <set>

namespace KINETO_NAMESPACE {

class CuptiActivityInterface {
 public:
  CuptiActivityInterface(const CuptiActivityInterface&) = delete;
  CuptiActivityInterface& operator=(const CuptiActivityInterface&) = delete;

  static CuptiActivityInterface& singleton();

  int smCount();
  static void pushCorrelationID(int id);
  static void popCorrelationID();

  void enableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void disableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  const std::pair<int, int> processActivities(
      std::function<void(const CUpti_Activity*)> handler);

  bool hasActivityBuffer() {
    return allocatedGpuBufferCount > 0;
  }

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection;
  int64_t flushOverhead{0};
  std::queue<std::pair<size_t, uint8_t*>> gpuTraceQueue;

 protected:
  CuptiActivityInterface() {}

 private:
  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const CUpti_Activity*)> handler);
  static void CUPTIAPI
  bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  static void CUPTIAPI bufferCompleted(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);

  int maxGpuBufferCount_{0};
  int allocatedGpuBufferCount{0};
};

} // namespace KINETO_NAMESPACE
