/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <CuptiRangeProfilerConfig.h>
#include <Logger.h>

#include <stdlib.h>

#include <fmt/format.h>
#include <ostream>


namespace KINETO_NAMESPACE {

// number of ranges affect the size of counter data binary used by
// the CUPTI Profiler. these defaults can be tuned
constexpr int KMaxAutoRanges = 1500; // supports 1500 kernels
constexpr int KMaxUserRanges = 10;   // enable upto 10 sub regions marked by user

constexpr char kCuptiProfilerMetricsKey[] = "CUPTI_PROFILER_METRICS";
constexpr char kCuptiProfilerPerKernelKey[] = "CUPTI_PROFILER_ENABLE_PER_KERNEL";
constexpr char kCuptiProfilerMaxRangesKey[] = "CUPTI_PROFILER_MAX_RANGES";

CuptiRangeProfilerConfig::CuptiRangeProfilerConfig(Config& cfg)
    : parent_(&cfg),
      cuptiProfilerPerKernel_(false),
      cuptiProfilerMaxRanges_(0) {}

bool CuptiRangeProfilerConfig::handleOption(const std::string& name, std::string& val) {
  VLOG(0) << " handling : " << name << " = " << val;
  // Cupti Range based Profiler configuration
  if (!name.compare(kCuptiProfilerMetricsKey)) {
    activitiesCuptiMetrics_ = splitAndTrim(val, ',');
  } else if (!name.compare(kCuptiProfilerPerKernelKey)) {
    cuptiProfilerPerKernel_ = toBool(val);
  } else if (!name.compare(kCuptiProfilerMaxRangesKey)) {
    cuptiProfilerMaxRanges_ = toInt64(val);
  } else {
    return false;
  }
  return true;
}

void CuptiRangeProfilerConfig::setDefaults() {
  if (activitiesCuptiMetrics_.size() > 0 && cuptiProfilerMaxRanges_ == 0) {
    cuptiProfilerMaxRanges_ =
      cuptiProfilerPerKernel_ ? KMaxAutoRanges : KMaxUserRanges;
  }
}

void CuptiRangeProfilerConfig::printActivityProfilerConfig(std::ostream& s) const {
  if (activitiesCuptiMetrics_.size() > 0) {
    s << "Cupti Profiler metrics : "
      << fmt::format("{}", fmt::join(activitiesCuptiMetrics_, ", ")) << std::endl;
    s << "Cupti Profiler measure per kernel : "
      << cuptiProfilerPerKernel_ << std::endl;
    s << "Cupti Profiler max ranges : " << cuptiProfilerMaxRanges_ << std::endl;
  }
}

void CuptiRangeProfilerConfig::registerFactory() {
  Config::addConfigFactory(
      kCuptiProfilerConfigName,
      [](Config& cfg) { return new CuptiRangeProfilerConfig(cfg); });
}

} // namespace KINETO_NAMESPACE
