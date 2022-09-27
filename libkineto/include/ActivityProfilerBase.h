#pragma once

#include <chrono>

#include "IActivityProfiler.h"
#include "ITraceActivity.h"

namespace libkineto {

class ActivityProfilerBase {

 public:
  virtual ~ActivityProfilerBase() = default;

  virtual bool isActive() const {
    fail();
    return false;
  }

  virtual const std::chrono::time_point<std::chrono::system_clock> performRunLoopStep(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime,
      int64_t currentIter = -1) {
    fail();
    return nextWakeupTime;
  }

  virtual void setLogger(ActivityLogger* logger) {
    fail();
  }

  virtual void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    fail();
  }

  virtual void stopTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    fail();
  }

  virtual void processTrace(ActivityLogger& logger) {
    fail();
  }

  virtual void reset() {
    fail();
  }

  virtual void configure(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    fail();
  }

  virtual void transferCpuTrace(
      std::unique_ptr<CpuTraceBuffer> cpuTrace) {
    fail();
  }

  virtual const Config& config() {
    fail();
    Config* config_;
    return *config_;
  }

  virtual inline void recordThreadInfo() {
    fail();
  }

  virtual void addMetadata(const std::string& key, const std::string& value) {
    fail();
  }

  virtual void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) {
    fail();
  }

 private:
  void fail() const {
    throw std::runtime_error("KINETO: ActivityProfilerBase class not instantiated.");
  }
};

}
