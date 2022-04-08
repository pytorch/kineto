// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>

#ifdef HAS_CUPTI
#include <cupti.h>
#endif

#include "ActivityType.h"
#include "CuptiActivityBuffer.h"


namespace KINETO_NAMESPACE {

using namespace libkineto;

#ifndef HAS_CUPTI
using CUpti_Activity = void;
#endif

class CuptiActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  CuptiActivityApi() = default;
  CuptiActivityApi(const CuptiActivityApi&) = delete;
  CuptiActivityApi& operator=(const CuptiActivityApi&) = delete;

  virtual ~CuptiActivityApi() {}

  static CuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void disableCuptiActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<CuptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      CuptiActivityBufferMap&,
      std::function<void(const CUpti_Activity*)> handler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

  static void forceLoadCupti();

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
  CuptiActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<CuptiActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};

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
