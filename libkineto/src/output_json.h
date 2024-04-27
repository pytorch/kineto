/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <ostream>
#include <ratio>
#include <thread>
#include <unordered_map>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "GenericTraceActivity.h"
#include "output_base.h"
#include "ActivityBuffers.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {
  // Previous declaration of TraceSpan is struct. Must match the same here.
  struct TraceSpan;
}

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleDeviceInfo(
      const DeviceInfo& info,
      uint64_t time) override;

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) override;

  std::string traceFileName() const {
    return fileName_;
  }

 protected:
  void finalizeTrace(
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata);

 private:

  // Create a flow event (arrow)
  void handleLink(
      char type,
      const ITraceActivity& e,
      int64_t id,
      const std::string& name);

  void addIterationMarker(const TraceSpan& span);

  void openTraceFile();

  void handleGenericInstantEvent(const ITraceActivity& op);

  void handleGenericLink(const ITraceActivity& activity);

  void metadataToJSON(
      const std::unordered_map<std::string, std::string>& metadata);

  void sanitizeStrForJSON(std::string& value);

  std::string fileName_;
  std::string tempFileName_;
  std::ofstream traceOf_;
};

//std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
# define _KINETO_GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
# define _KINETO_GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
# define _KINETO_GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

// There are tools like Chrome Trace Viewer that uses double to represent
// each element in the timeline. Double has a 53 bit mantissa to support
// up to 2^53 significant digits (up to 9007199254740992). This holds at the
// nanosecond level, about 3 months and 12 days. So, let's round base time to
// 3 months intervals, so we can still collect traces across ranks relative
// to each other.
// A month is 2629746, so 3 months is 7889238.
using _trimonths = std::chrono::duration<
    _KINETO_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
#undef _GLIBCXX_CHRONO_INT64_T

class ChromeTraceBaseTime {
 public:
  ChromeTraceBaseTime() = default;
  static ChromeTraceBaseTime& singleton();
  void init() {
    get();
  }
  int64_t get() {
    // Make all timestamps relative to 3 month intervals.
    static int64_t base_time = libkineto::timeSinceEpoch(
        std::chrono::time_point<std::chrono::system_clock>(
            std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
    return base_time;
  }
};

} // namespace KINETO_NAMESPACE
