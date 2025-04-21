#pragma once

#include <mutex>
#include <unordered_map>

#include "XpuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

class XpuptiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  XpuptiActivityProfilerSession() = delete;
  XpuptiActivityProfilerSession(
      XpuptiActivityApi& xpti,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);
  XpuptiActivityProfilerSession(const XpuptiActivityProfilerSession&) = delete;
  XpuptiActivityProfilerSession& operator=(
      const XpuptiActivityProfilerSession&) = delete;

  ~XpuptiActivityProfilerSession();

  void start() override;
  void stop() override;
  std::vector<std::string> errors() override {
    return errors_;
  };
  void processTrace(ActivityLogger& logger) override;
  void processTrace(
      ActivityLogger& logger,
      libkineto::getLinkedActivityCallback get_linked_activity,
      int64_t captureWindowStartTime,
      int64_t captureWindowEndTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

 private:
  void checkTimestampOrder(const ITraceActivity* act1);
  void removeCorrelatedPtiActivities(const ITraceActivity* act1);
  bool outOfRange(const ITraceActivity& act);
  int64_t getMappedQueueId(uint64_t sycl_queue_id);
  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);
  void handleCorrelationActivity(
      const pti_view_record_external_correlation* correlation);
  void handleRuntimeActivity(
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 10
      const pti_view_record_api* activity,
#else
      const pti_view_record_sycl_runtime* activity,
#endif
      ActivityLogger* logger);
  void handleKernelActivity(
      const pti_view_record_kernel* activity,
      ActivityLogger* logger);
  void handleMemcpyActivity(
      const pti_view_record_memory_copy* activity,
      ActivityLogger* logger);
  void handleMemsetActivity(
      const pti_view_record_memory_fill* activity,
      ActivityLogger* logger);
  void handleOverheadActivity(
      const pti_view_record_overhead* activity,
      ActivityLogger* logger);
  void handlePtiActivity(
      const pti_view_record_base* record,
      ActivityLogger* logger);

  // enumerate XPU Device UUIDs from runtime for once
  void enumDeviceUUIDs();

  // get logical device index(int8) from the given UUID from runtime
  // for profiling activity creation
  DeviceIndex_t getDeviceIdxFromUUID(const uint8_t deviceUUID[16]);

 private:
  static uint32_t iterationCount_;
  static std::vector<std::array<unsigned char, 16>> deviceUUIDs_;
  static std::vector<std::string> correlateRuntimeOps_;

  int64_t captureWindowStartTime_{0};
  int64_t captureWindowEndTime_{0};
  int64_t profilerStartTs_{0};
  int64_t profilerEndTs_{0};
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;
  std::unordered_map<int64_t, const ITraceActivity*> correlatedPtiActivities_;
  std::vector<std::string> errors_;

  libkineto::getLinkedActivityCallback cpuActivity_;

  XpuptiActivityApi& xpti_;
  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<uint64_t> sycl_queue_pool_;
  std::unique_ptr<const libkineto::Config> config_{nullptr};
  const std::set<ActivityType>& activity_types_;
};

class XPUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  XPUActivityProfiler() = default;
  XPUActivityProfiler(const XPUActivityProfiler&) = delete;
  XPUActivityProfiler& operator=(const XPUActivityProfiler&) = delete;

  const std::string& name() const override;
  const std::set<ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"__xpu_profiler__"};
  int64_t AsyncProfileStartTime_{0};
  int64_t AsyncProfileEndTime_{0};
};

} // namespace KINETO_NAMESPACE
