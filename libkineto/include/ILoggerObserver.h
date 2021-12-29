// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if !USE_GOOGLE_LOG

#include <map>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

enum LoggerOutputType {
  VERBOSE = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  ENUM_COUNT = 4
};

const char* toString(LoggerOutputType t);
LoggerOutputType toLoggerOutputType(const std::string& str);

constexpr int LoggerTypeCount = (int) LoggerOutputType::ENUM_COUNT;

class ILoggerObserver {
 public:
  virtual ~ILoggerObserver() = default;
  virtual void write(const std::string& message, LoggerOutputType ot) = 0;
  virtual const std::map<LoggerOutputType, std::vector<std::string>> extractCollectorMetadata() {
    return std::map<LoggerOutputType, std::vector<std::string>>();
  }

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
