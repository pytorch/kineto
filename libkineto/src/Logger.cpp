/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Logger.h"

#ifndef USE_GOOGLE_LOG

#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace KINETO_NAMESPACE {

int Logger::severityLevel_{VERBOSE};
int Logger::verboseLogLevel_{-1};
uint64_t Logger::verboseLogModules_{~0ull};

Logger::Logger(int severity, int line, const char* filePath, int errnum)
    : buf_(), out_(LIBKINETO_DBG_STREAM), errnum_(errnum) {
  switch (severity) {
    case INFO:
      buf_ << "INFO:";
      break;
    case WARNING:
      buf_ << "WARNING:";
      break;
    case ERROR:
      buf_ << "ERROR:";
      break;
    case VERBOSE:
      buf_ << "V:";
      break;
    default:
      buf_ << "???:";
      break;
  }

  const auto tt =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  const char* file = strrchr(filePath, '/');
  std::tm tm;
  buf_ << std::put_time(localtime_r(&tt, &tm), "%F %T") << " " << getpid()
       << ":" << syscall(SYS_gettid) << " " << (file ? file + 1 : filePath)
       << ":" << line << "] ";
}

Logger::~Logger() {
  if (errnum_ != 0) {
    thread_local char buf[1024];
    buf_ << " : " << strerror_r(errnum_, buf, sizeof(buf));
  }
  buf_ << std::ends;
  out_ << buf_.str() << std::endl;
}

void Logger::setVerboseLogModules(const std::vector<std::string>& modules) {
  if (modules.empty()) {
    verboseLogModules_ = ~0ull;
  } else {
    verboseLogModules_ = 0;
    for (const std::string& name : modules) {
      verboseLogModules_ |= hash(name.c_str());
    }
  }
}

} // namespace KINETO_NAMESPACE

#endif // USE_GOOGLE_LOG
