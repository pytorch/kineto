// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>
#include <fmt/format.h>

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

#define NVPW_CALL(call)                            \
  [&]() -> bool {                                  \
    NVPA_Status _status_ = call;                   \
    if (_status_ != NVPA_STATUS_SUCCESS) {         \
      LOG(WARNING) << fmt::format(                 \
          "function {} failed with error ({})",    \
          #call,                                   \
          (int)_status_);                          \
      return false;                                \
    }                                              \
    return true;                                   \
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
