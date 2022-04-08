// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

namespace KINETO_NAMESPACE {
  class Config;
}

namespace libkineto {

using namespace KINETO_NAMESPACE;

class ActivityLogger {
 public:

  virtual ~ActivityLogger() = default;

  struct DeviceInfo {
    DeviceInfo(int64_t id, const std::string& name, const std::string& label) :
      id(id), name(name), label(label) {}
    int64_t id;
    const std::string name;
    const std::string label;
  };

  struct ResourceInfo {
    ResourceInfo(
        int64_t deviceId,
        int64_t id,
        int64_t sortIndex,
        const std::string& name) :
        id(id), sortIndex(sortIndex), deviceId(deviceId), name(name) {}
    int64_t id;
    int64_t sortIndex;
    int64_t deviceId;
    const std::string name;
  };

  struct OverheadInfo {
    explicit OverheadInfo(const std::string& name) : name(name) {}
    const std::string name;
  };

  virtual void handleDeviceInfo(
      const DeviceInfo& info,
      uint64_t time) = 0;

  virtual void handleResourceInfo(const ResourceInfo& info, int64_t time) = 0;

  virtual void handleOverheadInfo(const OverheadInfo& info, int64_t time) = 0;

  virtual void handleTraceSpan(const TraceSpan& span) = 0;

  virtual void handleActivity(
      const libkineto::ITraceActivity& activity) = 0;
  virtual void handleGenericActivity(
      const libkineto::GenericTraceActivity& activity) = 0;

  virtual void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) = 0;

  void handleTraceStart() {
    handleTraceStart(std::unordered_map<std::string, std::string>());
  }

  virtual void finalizeTrace(
      const KINETO_NAMESPACE::Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) = 0;

 protected:
  ActivityLogger() = default;
};

} // namespace KINETO_NAMESPACE
