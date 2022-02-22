// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#ifdef HAS_CUPTI
#include <cupti.h>
#endif
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

  void handleGenericActivity(const ITraceActivity& activity) override;

#ifdef HAS_CUPTI
  void handleGpuActivity(const GpuActivity<CUpti_ActivityKernel4>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy2>& activity) override;
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemset>& activity) override;
#endif // HAS_CUPTI

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

  void metadataToJSON(const std::unordered_map<std::string, std::string>& metadata);

  std::string fileName_;
  std::ofstream traceOf_;
};

} // namespace KINETO_NAMESPACE
