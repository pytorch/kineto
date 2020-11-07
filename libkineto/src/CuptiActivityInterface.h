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
#include <cupti.h>
#include <functional>
#include <list>
#include <memory>
#include <set>

namespace KINETO_NAMESPACE {

using namespace libkineto;

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

  void addActivityBuffer(uint8_t* buffer, size_t validSize);
  std::unique_ptr<std::list<CuptiActivityBuffer>> activityBuffers();

  const std::pair<int, int> processActivities(
      std::list<CuptiActivityBuffer>& buffers,
      std::function<void(const CUpti_Activity*)> handler);

  bool hasActivityBuffer() {
    return allocatedGpuBufferCount > 0;
  }

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

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
  std::unique_ptr<std::list<CuptiActivityBuffer>> gpuTraceBuffers_;
};

} // namespace KINETO_NAMESPACE
