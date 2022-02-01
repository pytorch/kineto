// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "Logger.h"
#include "ILoggerObserver.h"

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
std::mutex Logger::loggerObserversMutex_;
#pragma GCC diagnostic pop


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

  {
    std::lock_guard<std::mutex> guard(loggerObserversMutex_);
    for (auto* observer : loggerObservers()) {
      // Output to observers. Current Severity helps keep track of which bucket the output goes.
      if (observer) {
        observer->write(buf_.str(), (LoggerOutputType) messageSeverity_);
      }
    }
  }

  // Finally, print to terminal or console.
  out_ << buf_.str() << std::endl;
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

void Logger::addLoggerObserver(ILoggerObserver* observer) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex_);
  loggerObservers().insert(observer);
}

void Logger::removeLoggerObserver(ILoggerObserver* observer) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex_);
  loggerObservers().erase(observer);
}

} // namespace KINETO_NAMESPACE

#endif // USE_GOOGLE_LOG
