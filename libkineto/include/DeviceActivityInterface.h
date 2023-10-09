#pragma once

#include <atomic>
#include <functional>
#include <set>
#include <stdint.h>

#include "ActivityType.h"
#include "ITraceActivity.h"

namespace libkineto {

class ActivityLogger;

class DeviceActivityInterface {
 public:
  virtual ~DeviceActivityInterface() {}

  virtual void pushCorrelationID(uint64_t id, int32_t type) = 0;
  virtual void popCorrelationID(int32_t type) = 0;

  virtual void enableActivities(const std::set<ActivityType>& selected_activities) = 0;
  virtual void disableActivities(const std::set<ActivityType>& selected_activities) = 0;
  virtual void clearActivities() = 0;
  virtual void teardownContext() = 0;
  virtual void setMaxBufferSize(int32_t size) = 0;

  virtual int32_t processActivities(ActivityLogger& logger,
                                    std::function<const ITraceActivity*(int32_t)> linked_activity,
                                    int64_t start_time, int64_t end_time) = 0;

 public:
  std::atomic_bool stopCollection{false};
};

extern DeviceActivityInterface* device_activity_singleton;

} // namespace libkineto