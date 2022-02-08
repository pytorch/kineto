// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <string>

#include "ActivityLoggerFactory.h"
#include "ActivityTraceInterface.h"
#include "output_json.h"
#include "output_membuf.h"

namespace libkineto {

class ActivityTrace : public ActivityTraceInterface {
 public:
  ActivityTrace(
      std::unique_ptr<MemoryTraceLogger> tmpLogger,
      const ActivityLoggerFactory& factory)
    : memLogger_(std::move(tmpLogger)),
      loggerFactory_(factory) {
  }

  const std::vector<const ITraceActivity*>* activities() override {
    return memLogger_->traceActivities();
  };

  void save(const std::string& url) override {
    std::string prefix;
    // if no protocol is specified, default to file
    if (url.find("://") == url.npos) {
      prefix = "file://";
    }
    memLogger_->log(*loggerFactory_.makeLogger(prefix + url));
  };

 private:
  // Activities are logged into a buffer
  std::unique_ptr<MemoryTraceLogger> memLogger_;

  // Alternative logger used by save() if protocol prefix is specified
  const ActivityLoggerFactory& loggerFactory_;
};

} // namespace libkineto
