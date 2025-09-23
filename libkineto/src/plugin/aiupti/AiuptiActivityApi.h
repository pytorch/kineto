#pragma once

#include "AiuptiActivityBuffer.h"
#include "AiuptiProfilerMacros.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <set>

namespace KINETO_NAMESPACE {

using Pti_Activity = AIUpti_Activity;

class AiuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  AiuptiActivityApi() = default;
  AiuptiActivityApi(const AiuptiActivityApi&) = delete;
  AiuptiActivityApi& operator=(const AiuptiActivityApi&) = delete;

  virtual ~AiuptiActivityApi() {}

  static AiuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableAiuptiActivities(
      const std::set<ActivityType>& selected_activities);
  void disablePtiActivities(const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<AiuptiActivityBufferDeque> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      AiuptiActivityBufferDeque&,
      std::function<void(const Pti_Activity*)> handler);

  void setMaxBufferSize(int size);
  // void setDeviceBufferSize(size_t size);
  // void setDeviceBufferPoolLimit(size_t limit);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxAiuBufferCount_{0};
  AiuptiActivityBufferDeque allocatedAiuTraceBuffers_;
  std::unique_ptr<AiuptiActivityBufferDeque> readyAiuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const Pti_Activity*)> handler);

  static void bufferRequestedTrampoline(
      uint8_t** buffer,
      size_t* size,
      size_t* maxNumRecords);
  static void
  bufferCompletedTrampoline(uint8_t* buffer, size_t size, size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

} // namespace KINETO_NAMESPACE
