// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if !USE_GOOGLE_LOG

#include <atomic>
#include <map>
#include <set>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace KINETO_NAMESPACE {

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    // Skip STAGE output type which is only used by USTLoggerCollector.
    if (ot != STAGE) {
      buckets_[ot].push_back(message);
    }
  }

  const std::map<LoggerOutputType, std::vector<std::string>> extractCollectorMetadata() override {
    return buckets_;
  }

  void addDevice(const int64_t device) override {
    devices.insert(device);
  }

  void setTraceDurationMS(const int64_t duration) override {
    trace_duration_ms = duration;
  }

  void addEventCount(const int64_t count) override {
    event_count += count;
  }

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

  // These are useful metadata to collect from CUPTIActivityProfiler for internal tracking.
  std::set<int64_t> devices;
  int64_t trace_duration_ms;
  std::atomic<uint64_t> event_count{0};

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
