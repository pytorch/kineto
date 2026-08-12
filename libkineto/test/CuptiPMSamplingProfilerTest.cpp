/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "include/Config.h"
#include "src/CuptiPMSamplingProfiler.h"
#include "src/CuptiTimestamp.h"
#include "src/ThrowUtil.h"
#include "src/output_membuf.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono_literals;

namespace {

struct FakeApiState {
  bool waitForFirstDecode() {
    std::unique_lock<std::mutex> lock(mutex);
    return decodeCondition.wait_for(
        lock, 2s, [this]() { return firstDecodeComplete; });
  }

  std::mutex mutex;
  std::condition_variable decodeCondition;
  CuptiPMSamplingConfig configured;
  std::vector<CuptiPMSample> firstBatch;
  int configureCalls{0};
  int startCalls{0};
  int decodeCalls{0};
  int stopCalls{0};
  int disableCalls{0};
  bool firstDecodeComplete{false};
  bool failConfigure{false};
};

class FakeCuptiPMSamplingApi final : public CuptiPMSamplingApi {
 public:
  explicit FakeCuptiPMSamplingApi(std::shared_ptr<FakeApiState> state)
      : state_(std::move(state)) {}

  void configure(const CuptiPMSamplingConfig& config) override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->configureCalls;
    state_->configured = config;
    if (state_->failConfigure) {
      KINETO_THROW(std::runtime_error, "configure failed");
    }
  }

  void start() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->startCalls;
  }

  bool decode(std::vector<CuptiPMSample>& samples) override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->decodeCalls;
    if (!state_->firstDecodeComplete) {
      samples.insert(
          samples.end(), state_->firstBatch.begin(), state_->firstBatch.end());
      state_->firstDecodeComplete = true;
      state_->decodeCondition.notify_all();
    }
    return true;
  }

  void stop() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->stopCalls;
  }

  void disable() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->disableCalls;
  }

 private:
  std::shared_ptr<FakeApiState> state_;
};

CuptiPMSamplingProfiler makeProfiler(
    const std::shared_ptr<FakeApiState>& state) {
  return CuptiPMSamplingProfiler(
      [state]() -> std::unique_ptr<CuptiPMSamplingApi> {
        return std::make_unique<FakeCuptiPMSamplingApi>(state);
      });
}

const std::set<ActivityType> kHardwareCounterActivities{
    ActivityType::HARDWARE_COUNTERS};

} // namespace

TEST(CuptiPMSamplingProfilerTest, ReportsNameAndSupportedActivity) {
  CuptiPMSamplingProfiler profiler;

  EXPECT_EQ(profiler.name(), "CUPTI PM Sampling");
  EXPECT_EQ(profiler.availableActivities(), kHardwareCounterActivities);
}

TEST(CuptiPMSamplingProfilerTest, RequiresActivityMetricsAndDevice) {
  auto state = std::make_shared<FakeApiState>();
  int apiCreations = 0;
  CuptiPMSamplingProfiler profiler(
      [state, &apiCreations]() -> std::unique_ptr<CuptiPMSamplingApi> {
        ++apiCreations;
        return std::make_unique<FakeCuptiPMSamplingApi>(state);
      });

  Config validConfig;
  ASSERT_TRUE(
      validConfig.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg\n"
                        "CUPTI_PM_SAMPLING_DEVICE_ID = 0"));
  EXPECT_EQ(profiler.configure({}, validConfig), nullptr);

  Config missingMetrics;
  ASSERT_TRUE(missingMetrics.parse("CUPTI_PM_SAMPLING_DEVICE_ID = 0"));
  EXPECT_EQ(
      profiler.configure(kHardwareCounterActivities, missingMetrics), nullptr);

  Config missingDevice;
  ASSERT_TRUE(
      missingDevice.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg"));
  EXPECT_EQ(
      profiler.configure(kHardwareCounterActivities, missingDevice), nullptr);

  EXPECT_EQ(apiCreations, 0);
}

TEST(CuptiPMSamplingProfilerTest, TimedConfigureForwardsSamplingConfig) {
  configureCuptiTimestampSource(false);
  auto state = std::make_shared<FakeApiState>();
  auto profiler = makeProfiler(state);

  Config config;
  ASSERT_TRUE(
      config.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg, "
                   "dram__bytes_read.sum\n"
                   "CUPTI_PM_SAMPLING_DEVICE_ID = 3"));

  auto session = profiler.configure(
      /*startTimeMs=*/123,
      /*durationMs=*/456,
      kHardwareCounterActivities,
      config);
  ASSERT_NE(session, nullptr);

  {
    std::lock_guard<std::mutex> lock(state->mutex);
    EXPECT_EQ(state->configureCalls, 1);
    EXPECT_EQ(state->configured.deviceId, 3);
    EXPECT_EQ(
        state->configured.metricNames,
        std::vector<std::string>(
            {"sm__cycles_active.avg", "dram__bytes_read.sum"}));
    EXPECT_EQ(state->configured.samplingInterval, 1ms);
  }

  session.reset();
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->disableCalls, 1);
}

TEST(CuptiPMSamplingProfilerTest, ReturnsNullWhenPreparationFails) {
  configureCuptiTimestampSource(false);
  auto state = std::make_shared<FakeApiState>();
  state->failConfigure = true;
  auto profiler = makeProfiler(state);

  Config config;
  ASSERT_TRUE(
      config.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg\n"
                   "CUPTI_PM_SAMPLING_DEVICE_ID = 0"));

  EXPECT_EQ(profiler.configure(kHardwareCounterActivities, config), nullptr);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->configureCalls, 1);
  EXPECT_EQ(state->disableCalls, 1);
}

TEST(CuptiPMSamplingProfilerTest, BuildsHardwareCounterActivities) {
  configureCuptiTimestampSource(false);
  auto state = std::make_shared<FakeApiState>();
  state->firstBatch = {
      CuptiPMSample{10, 20, {999.0, 999.0}},
      CuptiPMSample{100, 120, {1.25, 2048.0}},
      CuptiPMSample{130, 170, {2.5, 4096.0}},
  };
  auto profiler = makeProfiler(state);

  Config config;
  ASSERT_TRUE(
      config.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg, "
                   "dram__bytes_read.sum\n"
                   "CUPTI_PM_SAMPLING_DEVICE_ID = 2"));

  auto session = profiler.configure(kHardwareCounterActivities, config);
  ASSERT_NE(session, nullptr);
  session->start();
  ASSERT_TRUE(state->waitForFirstDecode());
  session->stop();

  {
    std::lock_guard<std::mutex> lock(state->mutex);
    EXPECT_EQ(state->startCalls, 1);
    EXPECT_GE(state->decodeCalls, 2);
    EXPECT_EQ(state->stopCalls, 1);
    EXPECT_EQ(state->disableCalls, 1);
  }

  MemoryTraceLogger logger(config);
  session->processTrace(logger);
  const auto* loggedActivities = logger.traceActivities();
  ASSERT_EQ(loggedActivities->size(), 2);

  const auto& first = *loggedActivities->at(0);
  EXPECT_EQ(first.type(), ActivityType::HARDWARE_COUNTERS);
  EXPECT_EQ(first.name(), "CUPTI PM Sampling");
  EXPECT_EQ(first.deviceId(), 2);
  EXPECT_EQ(first.timestamp(), 100);
  EXPECT_EQ(first.duration(), 20);
  ASSERT_EQ(first.counterValues().size(), 2);
  EXPECT_EQ(first.counterValues()[0].first, "sm__cycles_active.avg");
  EXPECT_DOUBLE_EQ(first.counterValues()[0].second, 1.25);
  EXPECT_EQ(first.counterValues()[1].first, "dram__bytes_read.sum");
  EXPECT_DOUBLE_EQ(first.counterValues()[1].second, 2048.0);

  const auto& second = *loggedActivities->at(1);
  EXPECT_EQ(second.timestamp(), 130);
  EXPECT_EQ(second.duration(), 40);
  ASSERT_EQ(second.counterValues().size(), 2);
  EXPECT_DOUBLE_EQ(second.counterValues()[0].second, 2.5);
  EXPECT_DOUBLE_EQ(second.counterValues()[1].second, 4096.0);

  auto buffer = session->getTraceBuffer();
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->span.name, "CUPTI PM Sampling");
  EXPECT_EQ(buffer->span.startTime, 100);
  EXPECT_EQ(buffer->span.endTime, 170);
  EXPECT_EQ(buffer->activities.size(), 2);
  EXPECT_EQ(session->getTraceBuffer().get(), nullptr);
}
