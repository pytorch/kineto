/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <string>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "Logger.h"

namespace KINETO_NAMESPACE {

struct CuptiRangeMeasurement {
  std::string rangeName;
  std::vector<double> values;
};

struct CuptiProfilerResult {
  std::vector<std::string> metricNames;
  // rangeName, list<double> values
  std::vector<CuptiRangeMeasurement> rangeVals;
};

/* Utilities for CUPTI and NVIDIA PerfWorks Metric API
 */

// ok to use fmt::format as error will not occur often. Can't use fmt::print
// easily since LOG(...) can return void, causes compiler error
#define NVPW_CALL(call)                                                \
  [&]() -> bool {                                                      \
    NVPA_Status _status_ = call;                                       \
    if (_status_ != NVPA_STATUS_SUCCESS) {                             \
      LOG(WARNING) << fmt::format(                                     \
          "function {} failed with error ({})", #call, (int)_status_); \
      return false;                                                    \
    }                                                                  \
    return true;                                                       \
  }()

// fixme - add a results string
// nvpperfGetResultString(_status_, &_errstr_);

namespace nvperf {

// Setup CUPTI profiler configuration blob and counter data image prefix
bool getProfilerConfigImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* counterAvailabilityImage = nullptr);

// Setup CUPTI profiler configuration blob and counter data image prefix
bool getCounterDataPrefixImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataImagePrefix);

/* NV Perf Metric Evaluation helpers
 *   - utilities to read binary data and obtain metrics for ranges
 */
CuptiProfilerResult evalMetricValues(
    const std::string& chipName,
    const std::vector<uint8_t>& counterDataImage,
    const std::vector<std::string>& metricNames,
    bool verbose = false);

} // namespace nvperf
} // namespace KINETO_NAMESPACE
