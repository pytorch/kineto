// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

#if !USE_GOOGLE_LOG

#include <array>
#include <fmt/format.h>

namespace KINETO_NAMESPACE {

struct LoggerTypeName {
  constexpr LoggerTypeName(const char* n, LoggerOutputType t) : name(n), type(t) {};
  const char* name;
  LoggerOutputType type;
};

static constexpr std::array<LoggerTypeName, LoggerTypeCount + 1> LoggerMap{{
    {"VERBOSE", LoggerOutputType::VERBOSE},
    {"INFO", LoggerOutputType::INFO},
    {"WARNING", LoggerOutputType::WARNING},
    {"ERROR", LoggerOutputType::ERROR},
    {"STAGE", LoggerOutputType::STAGE},
    {"???", LoggerOutputType::ENUM_COUNT}
}};

static constexpr bool matchingOrder(int idx = 0) {
  return LoggerMap[idx].type == LoggerOutputType::ENUM_COUNT ||
    ((idx == (int) LoggerMap[idx].type) && matchingOrder(idx + 1));
}
static_assert(matchingOrder(), "LoggerTypeName map is out of order");

const char* toString(LoggerOutputType t) {
  if(t < VERBOSE || t >= ENUM_COUNT) {
    return LoggerMap[ENUM_COUNT].name;
  }
  return LoggerMap[(int)t].name;
}

LoggerOutputType toLoggerOutputType(const std::string& str) {
  for (int i = 0; i < LoggerTypeCount; i++) {
    if (str == LoggerMap[i].name) {
      return LoggerMap[i].type;
    }
  }
  throw std::invalid_argument(fmt::format("Invalid activity type: {}", str));
}

} // namespace KINETO_NAMESPACE


#endif // !USE_GOOGLE_LOG
