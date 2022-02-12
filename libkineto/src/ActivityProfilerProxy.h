// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "ActivityProfilerInterface.h"

#include <memory>
#include <set>
#include <vector>

#include "ActivityType.h"
#include "ITraceActivity.h"

namespace libkineto {
  // previous declaration is struct so this one must be too.
  struct CpuTraceBuffer;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;

class ActivityProfilerController;
class Config;
class ConfigLoader;

class ActivityProfilerProxy : public ActivityProfilerInterface {

 public:
  ActivityProfilerProxy(bool cpuOnly, ConfigLoader& configLoader);
  ~ActivityProfilerProxy() override;

  void init() override;
  bool isInitialized() override {
    return controller_ != nullptr;
  }

  bool isActive() override;

  void recordThreadInfo() override;

  void scheduleTrace(const std::string& configStr) override;
  void scheduleTrace(const Config& config);

  void prepareTrace(
      const std::set<ActivityType>& activityTypes,
      const std::string& configStr = "") override;

  void startTrace() override;
  void step() override;
  std::unique_ptr<ActivityTraceInterface> stopTrace() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

  void transferCpuTrace(
     std::unique_ptr<CpuTraceBuffer> traceBuffer) override;

  void addMetadata(const std::string& key, const std::string& value) override;

  virtual void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) override;

 private:
  bool cpuOnly_{true};
  ConfigLoader& configLoader_;
  ActivityProfilerController* controller_{nullptr};
};

} // namespace libkineto
