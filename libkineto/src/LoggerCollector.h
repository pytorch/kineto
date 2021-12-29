// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if !USE_GOOGLE_LOG

#include <map>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace KINETO_NAMESPACE {

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    buckets_[ot].push_back(message);
  }

  const std::map<LoggerOutputType, std::vector<std::string>> extractCollectorMetadata() override {
    return buckets_;
  }

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
