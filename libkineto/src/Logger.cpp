// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "Logger.h"
#include "ILoggerObserver.h"
#include <atomic>
#include <math.h>

#ifndef USE_GOOGLE_LOG

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <time.h>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "ThreadUtil.h"

namespace KINETO_NAMESPACE {

std::atomic_int Logger::severityLevel_{VERBOSE};
std::atomic_int Logger::verboseLogLevel_{-1};
std::atomic<uint64_t> Logger::verboseLogModules_{~0ull};

// For iOS CI tests, skip error on global constructor without destructor here.
// This is valid C++ code, and shouldn't crash without destructor.
#if __clang__ && __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
// This is atomic, MUST ALWAYS use atomic_ functions for loggerObservers_.
std::shared_ptr<LoggerObserverList> Logger::loggerObservers_{nullptr};
#if __clang__ && __APPLE__
#pragma clang diagnostic pop
#endif

Logger::Logger(int severity, int line, const char* filePath, int errnum)
    : buf_(), out_(LIBKINETO_DBG_STREAM), errnum_(errnum), messageSeverity_(severity) {
  buf_ << toString((LoggerOutputType) severity) << ":";

  const auto tt =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  const char* file = strrchr(filePath, '/');
  buf_ << fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(tt)) << " "
       << processId() << ":" << systemThreadId() << " "
       << (file ? file + 1 : filePath) << ":" << line << "] ";
}

Logger::~Logger() {
#ifdef __linux__
  if (errnum_ != 0) {
    thread_local char buf[1024];
    buf_ << " : " << strerror_r(errnum_, buf, sizeof(buf));
  }
#endif

  std::string message = buf_.str();
  auto p = std::atomic_load(&loggerObservers_);
  if (p) {
    p->write_all(message, (LoggerOutputType) messageSeverity_);
  }

  // Finally, print to terminal or console.
  out_ << message << std::endl;
}

void Logger::setVerboseLogModules(const std::vector<std::string>& modules) {
  uint64_t mask = 0;
  if (modules.empty()) {
    mask = ~0ull;
  } else {
    for (const std::string& name : modules) {
      mask |= hash(name.c_str());
    }
  }
  verboseLogModules_ = mask;
}

void Logger::addLoggerObserver(std::shared_ptr<ILoggerObserver> observer) {
  auto p = std::atomic_load(&loggerObservers_);
  if (p) {
    p->push_front(observer);
  }
}

void Logger::removeLoggerObserver(std::shared_ptr<ILoggerObserver> observer) {
  auto p = std::atomic_load(&loggerObservers_);
  if (p) {
    p->remove(observer);
  }
}

} // namespace KINETO_NAMESPACE

#endif // USE_GOOGLE_LOG
