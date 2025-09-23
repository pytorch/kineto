#pragma once

#include <mutex>
#include <unordered_map>

#include "AiuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

class AiuptiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  AiuptiActivityProfilerSession() = delete;
  AiuptiActivityProfilerSession(
      AiuptiActivityApi& api,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);
  AiuptiActivityProfilerSession(const AiuptiActivityProfilerSession&) = delete;
  AiuptiActivityProfilerSession& operator=(
      const AiuptiActivityProfilerSession&) = delete;

  ~AiuptiActivityProfilerSession();

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
  void handleRuntimeActivity(
      const AIUpti_ActivityAPI* activity,
      ActivityLogger* logger);
  void handleKernelActivity(
      const AIUpti_ActivityCompute* activity,
      ActivityLogger* logger);
  void handleMemcpyActivity(
      const AIUpti_ActivityMemcpy* activity,
      ActivityLogger* logger);
  void handleMemsetActivity(
      const AIUpti_ActivityMemset* activity,
      ActivityLogger* logger);
  void handleMemoryActivity(
      const AIUpti_ActivityMemory* activity,
      ActivityLogger* logger);
  void handlePtiActivity(const AIUpti_Activity* record, ActivityLogger* logger);

  template <class memory_activity_type>
  uint32_t getResourceId(memory_activity_type* activity);

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
  std::map<std::pair<int64_t, int64_t>, std::vector<int64_t>> activeThreadMap_;
  std::vector<std::string> errors_;

  libkineto::getLinkedActivityCallback cpuActivity_;

  AiuptiActivityApi& api_;
  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<uint64_t> sycl_queue_pool_;
  std::unique_ptr<const libkineto::Config> config_{nullptr};
  const std::set<ActivityType>& activity_types_;

  // Ensures control block streams come after memory activities
  uint32_t kExceedMaxTid = 1000;

  std::map<std::pair<int64_t, int64_t>, ResourceInfo> resourceInfo_;
  bool hasDeviceResource(uint32_t device, uint32_t id);
  void recordStream(uint32_t device, uint32_t id);
  void recordMemoryStream(uint32_t device, uint32_t id, std::string kind);

  int64_t totalAllocatedBytes_{0};
};

class AIUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  AIUActivityProfiler() = default;
  AIUActivityProfiler(const AIUActivityProfiler&) = delete;
  AIUActivityProfiler& operator=(const AIUActivityProfiler&) = delete;

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
  std::string name_{"__aiu_profiler__"};
  int64_t AsyncProfileStartTime_{0};
  int64_t AsyncProfileEndTime_{0};
};

} // namespace KINETO_NAMESPACE
