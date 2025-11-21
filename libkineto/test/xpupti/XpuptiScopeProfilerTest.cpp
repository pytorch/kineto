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

template <class MAP, class EXPMAP>
void CheckCountsInMap(
    const MAP& map,
    unsigned expSize,
    unsigned repeatCount,
    const EXPMAP& expMap) {
  EXPECT_EQ(map.size(), expSize);

  std::map<unsigned, unsigned> countsMap;
  for (const auto& [key, val] : map) {
    countsMap[val]++;
  }

  EXPECT_EQ(countsMap.size(), expMap.size());

  for (auto itCountsMap = countsMap.begin(), itExpArray = expMap.begin();
       (itCountsMap != countsMap.end()) && (itExpArray != expMap.end());
       ++itCountsMap, ++itExpArray) {
    EXPECT_EQ(itCountsMap->first, itExpArray->first * repeatCount);
    EXPECT_EQ(itCountsMap->second, itExpArray->second);
  }
}

auto CountsMap(const std::vector<std::string>& names) {
  std::map<std::string, unsigned> counts;
  for (const auto& name : names) {
    counts[name]++;
  }

  std::map<unsigned, unsigned> countsMap;
  for (const auto& [name, count] : counts) {
    countsMap[count]++;
  }

  return std::pair{countsMap, counts.size()};
};

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

    int64_t startTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    pSession->start();

    constexpr unsigned repeatCount = 5;
    void ComputeOnXpu(unsigned size, unsigned repeatCount);
    ComputeOnXpu(1024, repeatCount);

    pSession->stop();
    int64_t endTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

    auto getLinkedActivity = [](int32_t) -> const KN::ITraceActivity* {
      return nullptr;
    };

    TestActivityLogger logger;
    pSession->processTrace(logger, getLinkedActivity, startTime, endTime);

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

    const std::vector<std::string> expectedActivities = {
        "urEnqueueMemBufferWrite",
        "urEnqueueMemBufferWrite",
        "urEnqueueMemBufferWrite",
        "Memcpy M2D",
        "Memcpy M2D",
        "Memcpy M2D",
        "urEnqueueKernelLaunch",
        "Run(sycl::_V1::queue, ...)",
        "urEnqueueMemBufferRead",
        "Memcpy D2M",
        "metrics: Run(sycl::_V1::queue, ...)",
        "metrics",
        "metrics"};

    auto [expectedActivitiesCountsMap, numUniqueActivities] =
        CountsMap(expectedActivities);

    const unsigned numMetrics = std::count_if(
        expectedActivities.begin(),
        expectedActivities.end(),
        [](const std::string& name) { return name.find("metrics") == 0; });

    const std::vector<std::string> expectedTypes = {
        "xpu_runtime",
        "xpu_runtime",
        "xpu_runtime",
        "gpu_memcpy",
        "gpu_memcpy",
        "gpu_memcpy",
        "xpu_runtime",
        "kernel",
        "xpu_runtime",
        "gpu_memcpy",
        "kernel",
        "xpu_scope_profiler",
        "xpu_scope_profiler"};

    auto [expectedTypesCountsMap, numUniqueTypes] = CountsMap(expectedTypes);

    const unsigned numActivities = expectedActivities.size();
    EXPECT_EQ(numActivities, expectedTypes.size());

    EXPECT_EQ(pBuffer->activities.size(), numActivities * repeatCount);

    std::map<std::string, unsigned> activitiesCount;
    std::map<KN::ActivityType, unsigned> typesCount;
    unsigned scopeProfilerActCount = 0;

    for (auto&& pActivity : pBuffer->activities) {
      activitiesCount[pActivity->name()]++;
      typesCount[pActivity->type()]++;

      bool isNameMetrics = pActivity->name() == "metrics";
      bool nameStartsWithMetrics = pActivity->name().find("metrics:") == 0;
      auto [metricsCount, metricsMask] =
          CountMetricsInString(metrics, pActivity->metadataJson());

      switch (pActivity->type()) {
        case KN::ActivityType::CONCURRENT_KERNEL:
          if (nameStartsWithMetrics)
            goto label_scope;
          else
            goto label_default;

        case KN::ActivityType::XPU_SCOPE_PROFILER:
          EXPECT_TRUE(isNameMetrics);
        label_scope:
          EXPECT_EQ(metricsCount, metrics.size());
          EXPECT_EQ(metricsMask, (1u << metrics.size()) - 1);
          ++scopeProfilerActCount;
          break;

        default:
        label_default:
          EXPECT_FALSE(isNameMetrics);
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

    EXPECT_EQ(scopeProfilerActCount, numMetrics * repeatCount);

    CheckCountsInMap(
        activitiesCount,
        numUniqueActivities,
        repeatCount,
        expectedActivitiesCountsMap);

    CheckCountsInMap(
        typesCount, numUniqueTypes, repeatCount, expectedTypesCountsMap);
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
