/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Logger.h>
#include <XpuptiScopeProfilerConfig.h>

#include <stdlib.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/scopes.h>
#include <ostream>

namespace KINETO_NAMESPACE {

// number of scopes affect the size of counter data binary used by
// the XPUPTI Profiler. these defaults can be tuned
constexpr int KMaxAutoScopes = 1500; // supports 1500 kernels
constexpr int KMaxUserScopes = 10; // enable upto 10 sub regions marked by user

constexpr char kXpuptiProfilerMetricsKey[] = "XPUPTI_PROFILER_METRICS";
constexpr char kXpuptiProfilerPerKernelKey[] =
    "XPUPTI_PROFILER_ENABLE_PER_KERNEL";
constexpr char kXpuptiProfilerMaxScopesKey[] = "XPUPTI_PROFILER_MAX_SCOPES";

bool XpuptiScopeProfilerConfig::handleOption(
    const std::string& name,
    std::string& val) {
  VLOG(0) << " handling : " << name << " = " << val;
  // Xpupti Scope based Profiler configuration
  if (!name.compare(kXpuptiProfilerMetricsKey)) {
    activitiesXpuptiMetrics_ = splitAndTrim(val, ',');
  } else if (!name.compare(kXpuptiProfilerPerKernelKey)) {
    xpuptiProfilerPerKernel_ = toBool(val);
  } else if (!name.compare(kXpuptiProfilerMaxScopesKey)) {
    xpuptiProfilerMaxScopes_ = toInt64(val);
  } else {
    return false;
  }
  return true;
}

void XpuptiScopeProfilerConfig::setDefaults() {
  if (activitiesXpuptiMetrics_.size() > 0 && xpuptiProfilerMaxScopes_ == 0) {
    xpuptiProfilerMaxScopes_ =
        xpuptiProfilerPerKernel_ ? KMaxAutoScopes : KMaxUserScopes;
  }
}

void XpuptiScopeProfilerConfig::printActivityProfilerConfig(
    std::ostream& s) const {
  if (activitiesXpuptiMetrics_.size() > 0) {
    fmt::print(
        s,
        "Xpupti Profiler metrics : {}\n"
        "Xpupti Profiler measure per kernel : {}\n"
        "Xpupti Profiler max scopes : {}\n",
        fmt::join(activitiesXpuptiMetrics_, ", "),
        xpuptiProfilerPerKernel_,
        xpuptiProfilerMaxScopes_);
  }
}

void XpuptiScopeProfilerConfig::registerFactory() {
  Config::addConfigFactory(kXpuptiProfilerConfigName, [](Config& cfg) {
    return new XpuptiScopeProfilerConfig(cfg);
  });
}

} // namespace KINETO_NAMESPACE
