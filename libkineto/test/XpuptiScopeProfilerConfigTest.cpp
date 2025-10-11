/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/plugin/xpupti/XpuptiScopeProfilerConfig.h"
#include "include/Config.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <time.h>
#include <chrono>

using namespace KINETO_NAMESPACE;

class XpuptiScopeProfilerConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    XpuptiScopeProfilerConfig::registerFactory();
  }
};

TEST_F(XpuptiScopeProfilerConfigTest, ConfigureProfiler) {
  Config cfg;
  std::vector<std::string> metrics = {
      "metric1",
      "metric2",
      "metric3",
  };
  auto metricsConfigStr =
      fmt::format("XPUPTI_PROFILER_METRICS = {}", fmt::join(metrics, ","));

  EXPECT_TRUE(cfg.parse(metricsConfigStr));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = true"));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_MAX_SCOPES = 314159"));

  const XpuptiScopeProfilerConfig& xpupti_cfg =
      XpuptiScopeProfilerConfig::get(cfg);

  EXPECT_EQ(xpupti_cfg.activitiesXpuptiMetrics(), metrics);
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerPerKernel(), true);
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerMaxScopes(), 314159);
}

TEST_F(XpuptiScopeProfilerConfigTest, RangesDefaults) {
  Config cfg, cfg_auto;

  // do not set max ranges in config, check defaults are sane
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_METRICS = metric1"));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = false"));

  cfg.setSignalDefaults();

  EXPECT_TRUE(cfg_auto.parse("XPUPTI_PROFILER_METRICS = metric2"));
  EXPECT_TRUE(cfg_auto.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = true"));

  cfg_auto.setClientDefaults();

  int user_ranges, auto_ranges;

  user_ranges = XpuptiScopeProfilerConfig::get(cfg).xpuptiProfilerMaxScopes();
  auto_ranges =
      XpuptiScopeProfilerConfig::get(cfg_auto).xpuptiProfilerMaxScopes();

  EXPECT_EQ(user_ranges, 10)
  EXPECT_EQ(auto_ranges, 1500)
}
