#pragma once

#include "XpuptiActivityBuffer.h"
#include "XpuptiProfilerMacros.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <set>

namespace KINETO_NAMESPACE {

using Pti_Activity = pti_view_record_base;

class XpuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  XpuptiActivityApi() = default;
  XpuptiActivityApi(const XpuptiActivityApi&) = delete;
  XpuptiActivityApi& operator=(const XpuptiActivityApi&) = delete;

  virtual ~XpuptiActivityApi() {}

  static XpuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableXpuptiActivities(
      const std::set<ActivityType>& selected_activities);
  void disablePtiActivities(const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<XpuptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      XpuptiActivityBufferMap&,
      std::function<void(const Pti_Activity*)> handler);

  void setMaxBufferSize(int size);
  // void setDeviceBufferSize(size_t size);
  // void setDeviceBufferPoolLimit(size_t limit);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxGpuBufferCount_{0};
  XpuptiActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XpuptiActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const Pti_Activity*)> handler);
  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size);
  static void
  bufferCompletedTrampoline(uint8_t* buffer, size_t size, size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

} // namespace KINETO_NAMESPACE
