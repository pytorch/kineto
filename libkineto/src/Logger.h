/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * glog has a couple of big issues:
 *  1) It crashes or terminates when linked both statically and dynamically
 *  2) VLOG before init crashes - this is a problem because parts of libkineto
 *     may be initialized before main()
 *
 * For these reasons we use our own implementations of glog macros
 * that just log to stderr by default.
 *
 */

#pragma once

#include <iostream>

#define LIBKINETO_DBG_STREAM std::cerr

#if USE_GOOGLE_LOG

#include <glog/logging.h>

#define SET_LOG_SEVERITY_LEVEL(level)
#define SET_LOG_VERBOSITY_LEVEL(level, modules)

#else // !USE_GOOGLE_LOG
#include <stdio.h>
#include <atomic>
#include <ostream>
#include <string>
#include <sstream>
#include <vector>

// unset a predefined ERROR (windows)
#undef ERROR

#define VERBOSE 0
#define INFO 1
#define WARNING 2
#define ERROR 3

namespace libkineto {

class Logger {
 public:
  Logger(int severity, int line, const char* filePath, int errnum = 0);
  ~Logger();

  inline std::ostream& stream() {
    return buf_;
  }

  static inline void setSeverityLevel(int level) {
    severityLevel_ = level;
  }

  static inline int severityLevel() {
    return severityLevel_;
  }

  static inline void setVerboseLogLevel(int level) {
    verboseLogLevel_ = level;
  }

  static inline int verboseLogLevel() {
    return verboseLogLevel_;
  }

  // This is constexpr so that the hash for a file name is computed at compile
  // time when used in the VLOG macros.
  // This way, there is no string comparison for matching VLOG modules,
  // only a comparison of pre-computed hashes.
  // No fancy hashing needed here. It's pretty inefficient (one character
  // at a time) but the strings are not large and it's not in the critical path.
  static constexpr uint64_t rol(uint64_t val, int amount) {
    return val << amount | val >> (63 - amount);
  }
  static constexpr uint64_t hash(const char* s) {
    uint64_t hash = hash_rec(s, 0);
    return hash & rol(0x41a0240682483014ull, hash & 63);
  }
  static constexpr uint64_t hash_rec(const char* s, int off) {
    // Random constants!
    return (!s[off] ? 57ull : (hash_rec(s, off + 1) * 293) ^ s[off]);
  }
  static constexpr const char* basename(const char* s, int off = 0) {
    return !s[off]
        ? s
        : s[off] == '/' ? basename(&s[off + 1]) : basename(s, off + 1);
  }

  static void setVerboseLogModules(const std::vector<std::string>& modules);

  static inline uint64_t verboseLogModules() {
    return verboseLogModules_;
  }

 private:
  std::stringstream buf_;
  std::ostream& out_;
  int errnum_;
  static std::atomic_int severityLevel_;
  static std::atomic_int verboseLogLevel_;
  static std::atomic<uint64_t> verboseLogModules_;
};

class VoidLogger {
 public:
  VoidLogger() {}
  void operator&(std::ostream&) {}
};

} // namespace libkineto

#ifdef LOG // Undefine in case these are already defined (quite likely)
#undef LOG
#undef LOG_IS_ON
#undef LOG_IF
#undef LOG_EVERY_N
#undef LOG_IF_EVERY_N
#undef DLOG
#undef DLOG_IF
#undef VLOG
#undef VLOG_IF
#undef VLOG_EVERY_N
#undef VLOG_IS_ON
#undef DVLOG
#undef LOG_FIRST_N
#undef CHECK
#undef DCHECK
#undef DCHECK_EQ
#undef PLOG
#undef PCHECK
#undef LOG_OCCURRENCES
#endif

#define LOG_IS_ON(severity) \
  (severity >= libkineto::Logger::severityLevel())

#define LOG_IF(severity, condition) \
  !(LOG_IS_ON(severity) && (condition)) ? (void)0 : libkineto::VoidLogger() & \
    libkineto::Logger(severity, __LINE__, __FILE__).stream()

#define LOG(severity) LOG_IF(severity, true)

#define LOCAL_VARNAME_CONCAT(name, suffix) _##name##suffix##_

#define LOCAL_VARNAME(name) LOCAL_VARNAME_CONCAT(name, __LINE__)

#define LOG_OCCURRENCES LOCAL_VARNAME(log_count)

#define LOG_EVERY_N(severity, rate)               \
  static int LOG_OCCURRENCES = 0;                 \
  LOG_IF(severity, LOG_OCCURRENCES++ % rate == 0) \
      << "(x" << LOG_OCCURRENCES << ") "

template <uint64_t n>
struct __to_constant__ {
  static const uint64_t val = n;
};
#define FILENAME_HASH                             \
  __to_constant__<libkineto::Logger::hash( \
      libkineto::Logger::basename(__FILE__))>::val
#define VLOG_IS_ON(verbosity)                           \
  (libkineto::Logger::verboseLogLevel() >= verbosity && \
   (libkineto::Logger::verboseLogModules() & FILENAME_HASH) == FILENAME_HASH)

#define VLOG_IF(verbosity, condition) \
  LOG_IF(VERBOSE, VLOG_IS_ON(verbosity) && (condition))

#define VLOG(verbosity) VLOG_IF(verbosity, true)

#define VLOG_EVERY_N(verbosity, rate)               \
  static int LOG_OCCURRENCES = 0;                   \
  VLOG_IF(verbosity, LOG_OCCURRENCES++ % rate == 0) \
      << "(x" << LOG_OCCURRENCES << ") "

#define PLOG(severity) \
  libkineto::Logger(severity, __LINE__, __FILE__, errno).stream()

#define SET_LOG_SEVERITY_LEVEL(level) \
  libkineto::Logger::setSeverityLevel(level)

#define SET_LOG_VERBOSITY_LEVEL(level, modules)   \
  libkineto::Logger::setVerboseLogLevel(level); \
  libkineto::Logger::setVerboseLogModules(modules)

#endif // USE_GOOGLE_LOG
