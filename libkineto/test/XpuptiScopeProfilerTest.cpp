/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/Config.h"
#include "include/output_base.h"

#include "src/plugin/xpupti/XpuptiActivityProfiler.h"
#include "src/plugin/xpupti/XpuptiScopeProfilerConfig.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

namespace KN = KINETO_NAMESPACE;

class XpuptiScopeProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KN::XpuptiScopeProfilerConfig::registerFactory();
  }
};

class TestActivityLogger : public KN::ActivityLogger {
  void handleDeviceInfo(const KN::DeviceInfo& info, uint64_t time) override {}
  void handleResourceInfo(const KN::ResourceInfo& info, int64_t time) override {
  }
  void handleOverheadInfo(
      const KN::ActivityLogger::OverheadInfo& info,
      int64_t time) override {}
  void handleTraceSpan(const KN::TraceSpan& span) override {}
  void handleActivity(const KN::ITraceActivity& activity) override {}
  void handleGenericActivity(
      const KN::GenericTraceActivity& activity) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override {}
  void finalizeMemoryTrace(const std::string&, const KN::Config&) override {}
  void finalizeTrace(
      const KN::Config& config,
      std::unique_ptr<KN::ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata)
      override {}
};

void RunTest(std::string_view perKernel, unsigned maxScopes) {
  KN::Config cfg;

  std::vector<std::string_view> metrics = {
      "GpuTime",
      "GpuCoreClocks",
      "AvgGpuCoreFrequencyMHz",
      "XVE_BUSY",
      "XVE_ACTIVE",
      "XVE_STALL"};

  EXPECT_TRUE(cfg.parse(
      fmt::format("XPUPTI_PROFILER_METRICS = {}", fmt::join(metrics, ","))));
  EXPECT_TRUE(cfg.parse(
      fmt::format("XPUPTI_PROFILER_ENABLE_PER_KERNEL = {}", perKernel)));
  EXPECT_TRUE(
      cfg.parse(fmt::format("XPUPTI_PROFILER_MAX_SCOPES = {}", maxScopes)));

  KN::XPUActivityProfiler profiler;
  EXPECT_TRUE(profiler.name() == "__xpu_profiler__");

  std::set<KN::ActivityType> activities{
      KN::ActivityType::GPU_MEMCPY,
      KN::ActivityType::GPU_MEMSET,
      KN::ActivityType::CONCURRENT_KERNEL,
      KN::ActivityType::EXTERNAL_CORRELATION,
      KN::ActivityType::XPU_RUNTIME,
      KN::ActivityType::XPU_SCOPE_PROFILER,
      KN::ActivityType::OVERHEAD};

  auto pSession = profiler.configure(activities, cfg);

  pSession->start();

  pSession->stop();

  TestActivityLogger logger;
  pSession->processTrace(logger);

  EXPECT_TRUE(pSession->errors().empty())
      << fmt::format("{}", fmt::join(pSession->errors(), ","));
}

TEST_F(XpuptiScopeProfilerTest, PerKernelScope) {
  RunTest("true", 314);
}

TEST_F(XpuptiScopeProfilerTest, UserScope) {
  RunTest("false", 159);
}
