// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {
  // Previous declaration of TraceSpan is struct. Must match the same here.
  struct TraceSpan;
}

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleDeviceInfo(
      const DeviceInfo& info,
      uint64_t time) override;

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) override;

  std::string traceFileName() const {
    return fileName_;
  }

 protected:
  void finalizeTraceInternal(
  int64_t endTime,
  std::unordered_map<std::string, std::vector<std::string>>& metadata);

 private:

  // Create a flow event (arrow)
  void handleLink(
      char type,
      const ITraceActivity& e,
      int64_t id,
      const std::string& cat,
      const std::string& name);

  void addIterationMarker(const TraceSpan& span);

  void openTraceFile();

  void handleGenericInstantEvent(const ITraceActivity& op);

  void handleGenericLink(const ITraceActivity& activity);

  void metadataToJSON(
      const std::unordered_map<std::string, std::string>& metadata);

  std::string& sanitizeStrForJSON(std::string& value);

  std::string fileName_;
  std::ofstream traceOf_;
};

} // namespace KINETO_NAMESPACE
