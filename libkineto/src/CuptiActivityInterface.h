/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>
#include <queue>

namespace KINETO_NAMESPACE {

class CuptiActivityInterface {
 public:
  CuptiActivityInterface(const CuptiActivityInterface&) = delete;
  CuptiActivityInterface& operator=(const CuptiActivityInterface&) = delete;

  static CuptiActivityInterface& singleton();

  int smCount();
  static void pushCorrelationID(int id);
  static void popCorrelationID();

  void enableCuptiActivities();
  void disableCuptiActivities();
  void flushActivities();

  // TODO: Replace with handler-based approach
  const CUpti_Activity* nextActivityRecord(uint8_t* buffer, size_t valid_size);

  void setMaxGpuBufferSize(int size);

  int allocatedGpuBufferCount = 0;
  bool stopCollection = false;
  std::queue<std::pair<size_t, uint8_t*>> gpuTraceQueue;

 protected:
  CuptiActivityInterface() {}

 private:
  static void CUPTIAPI
  bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  static void CUPTIAPI bufferCompleted(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);

  int maxGpuBufferCount_ = 0;
  CUpti_Activity* currentRecord_ = nullptr;
};

} // namespace KINETO_NAMESPACE
