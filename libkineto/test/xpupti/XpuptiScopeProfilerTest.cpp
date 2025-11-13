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

bool IsEnvVerbose() {
  auto verboseEnv = getenv("VERBOSE");
  return verboseEnv && (strcmp(verboseEnv, "1") == 0);
}

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

std::ostream& operator<<(std::ostream& os, const KN::TraceSpan& span) {
#define PRINT(V) os << std::endl << "      " #V " = " << span.V;
  PRINT(startTime)
  PRINT(endTime)
  PRINT(opCount)
  PRINT(iteration)
  PRINT(name)
  PRINT(prefix)
#undef PRINT
  return os;
}

std::ostream& operator<<(std::ostream& os, KN::ActivityType actType) {
  os << toString(actType);
  return os;
}

auto CountMetricsInString(
    const std::vector<std::string_view>& metrics,
    const std::string_view sv) {
  unsigned metricsCount = 0;
  unsigned metricsMask = 0;
  for (unsigned i = 0; i < metrics.size(); ++i) {
    const auto metricSv = metrics[i];
    if (sv.find(metricSv) != std::string::npos) {
      ++metricsCount;
      metricsMask |= (1 << i);
    }
  }
  return std::pair{metricsCount, metricsMask};
}

template <class MAP, class ARRAY>
void CheckCountsInMap(
    const MAP& map,
    unsigned expSize,
    unsigned repeatCount,
    const ARRAY& expArray) {
  EXPECT_EQ(map.size(), expSize);

  std::map<unsigned, unsigned> countsMap;
  for (const auto& [key, val] : map) {
    countsMap[val]++;
  }

  EXPECT_EQ(countsMap.size(), expArray.size());

  for (auto itCountsMap = countsMap.begin(), itExpArray = expArray.begin();
       (itCountsMap != countsMap.end()) && (itExpArray != expArray.end());
       ++itCountsMap, ++itExpArray) {
    EXPECT_EQ(itCountsMap->first, itExpArray->first * repeatCount);
    EXPECT_EQ(itCountsMap->second, itExpArray->second);
  }
}

void RunTest(std::string_view perKernel, unsigned maxScopes) {
  KN::Config cfg;

  std::vector<std::string_view> metrics = {
      "GpuTime",
      "GpuCoreClocks",
      "AvgGpuCoreFrequencyMHz",
      "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
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

#if PTI_VERSION_AT_LEAST(0, 14)
  if (perKernel == "false") {
    EXPECT_THROW(profiler.configure(activities, cfg), std::runtime_error);
  } else {
    auto pSession = profiler.configure(activities, cfg);

    pSession->start();

    constexpr unsigned repeatCount = 5;
    void ComputeOnXpu(unsigned size, unsigned repeatCount);
    ComputeOnXpu(1024, repeatCount);

    pSession->stop();

    TestActivityLogger logger;
    pSession->processTrace(logger);

    EXPECT_TRUE(pSession->errors().empty())
        << fmt::format("{}", fmt::join(pSession->errors(), ","));

    auto pBuffer = pSession->getTraceBuffer();

    static bool isVerbose = IsEnvVerbose();

    if (isVerbose) {
      std::cout << "span = " << pBuffer->span << std::endl;
      std::cout << "gpuOpCount = " << pBuffer->gpuOpCount << std::endl;
      std::cout << "activities.size = " << pBuffer->activities.size()
                << std::endl;
    }

    constexpr unsigned numMemWrites = 3;
    constexpr unsigned numMemWritesCount = 2;
    constexpr unsigned numMemReads = 1;
    constexpr unsigned numMemReadsCount = 2;
    constexpr unsigned numKernels = 1;
    constexpr unsigned numKernelsCount = 3;
    constexpr unsigned numActivities = numMemWrites * numMemWritesCount +
        numMemReads * numMemReadsCount + numKernels * numKernelsCount;

    EXPECT_EQ(pBuffer->activities.size(), repeatCount * numActivities);

    std::map<std::string, unsigned> activitiesCount;
    std::map<KN::ActivityType, unsigned> typesCount;
    unsigned scopeProfilerActCount = 0;

    for (auto&& pActivity : pBuffer->activities) {
      activitiesCount[pActivity->name()]++;
      typesCount[pActivity->type()]++;

      auto posMetricsStr = pActivity->name().find("metrics: ");
      auto [metricsCount, metricsMask] =
          CountMetricsInString(metrics, pActivity->metadataJson());

      if (pActivity->type() == KN::ActivityType::XPU_SCOPE_PROFILER) {
        EXPECT_EQ(posMetricsStr, 0);
        EXPECT_EQ(metricsCount, metrics.size());
        EXPECT_EQ(metricsMask, (1u << metrics.size()) - 1);
        ++scopeProfilerActCount;
      } else {
        EXPECT_EQ(posMetricsStr, std::string::npos);
        EXPECT_EQ(metricsCount, 0);
        EXPECT_EQ(metricsMask, 0);
      }

      if (isVerbose) {
#define PRINT(A) std::cout << #A " = " << pActivity->A() << std::endl;
        PRINT(deviceId)
        PRINT(resourceId)
        PRINT(getThreadId)
        PRINT(timestamp)
        PRINT(duration)
        PRINT(correlationId)
        PRINT(linkedActivity)
        PRINT(flowType)
        PRINT(flowId)
        PRINT(flowStart)
        PRINT(name)
        PRINT(type)
        PRINT(metadataJson)
#undef PRINT

        if (auto pSpan = pActivity->traceSpan()) {
          std::cout << "traceSpan = " << *pSpan << std::endl;
        }

        std::cout << "-----" << std::endl;
      }
    }

    EXPECT_EQ(scopeProfilerActCount, repeatCount);

    CheckCountsInMap(
        activitiesCount,
        7,
        repeatCount,
        std::array{std::pair{1u, 5}, std::pair{numMemWrites, 2}});

    CheckCountsInMap(
        typesCount,
        4,
        repeatCount,
        std::array{
            std::pair{1u, 2},
            std::pair{numMemWrites + numMemReads, 1},
            std::pair{numMemWrites + numMemReads + numKernels, 1}});
  }
#else
  EXPECT_THROW(profiler.configure(activities, cfg), std::runtime_error);
#endif
}

TEST_F(XpuptiScopeProfilerTest, PerKernelScope) {
  RunTest("true", 314);
}

TEST_F(XpuptiScopeProfilerTest, UserScope) {
  RunTest("false", 159);
}
