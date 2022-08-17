/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/EventProfiler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <time.h>

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

TEST(PercentileTest, Create) {
  PercentileList pct = {{10, SampleValue(0)},
                        {49, SampleValue(0)},
                        {50, SampleValue(0)},
                        {90, SampleValue(0)}};

  percentiles<int>({0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, pct);
  EXPECT_EQ(pct[0].second.getInt(), 10);
  EXPECT_EQ(pct[1].second.getInt(), 50);
  EXPECT_EQ(pct[2].second.getInt(), 50);
  EXPECT_EQ(pct[3].second.getInt(), 90);

  percentiles<int>({80, 10, 20, 70, 60, 40, 90, 30, 50, 0, 100}, pct);
  EXPECT_EQ(pct[0].second.getInt(), 10);
  EXPECT_EQ(pct[1].second.getInt(), 50);
  EXPECT_EQ(pct[2].second.getInt(), 50);
  EXPECT_EQ(pct[3].second.getInt(), 90);

  percentiles<int>({80}, pct);
  EXPECT_EQ(pct[0].second.getInt(), 80);
  EXPECT_EQ(pct[1].second.getInt(), 80);
  EXPECT_EQ(pct[2].second.getInt(), 80);
  EXPECT_EQ(pct[3].second.getInt(), 80);

  percentiles<int>({80, 50}, pct);
  EXPECT_EQ(pct[0].second.getInt(), 50);
  EXPECT_EQ(pct[1].second.getInt(), 50);
  EXPECT_EQ(pct[2].second.getInt(), 80);
  EXPECT_EQ(pct[3].second.getInt(), 80);
}

TEST(PercentileTest, Normalize) {
  PercentileList pct = {
      {10, SampleValue(10)}, {50, SampleValue(100.0)}, {90, SampleValue(2000)}};

  normalize(pct, 2.5);

  EXPECT_EQ(pct[0].second.getInt(), 25);
  EXPECT_EQ((int)pct[1].second.getDouble(), 250);
  EXPECT_EQ(pct[2].second.getInt(), 5000);
}

TEST(EventTest, SumSamples) {
  Event ev;
  ev.instanceCount = 4;
  auto t = system_clock::now();
  ev.addSample(t, {1, 2, 3, 4});
  ev.addSample(t, {10, 20, 30, 40});
  ev.addSample(t, {100, 200, 300, 400});

  EXPECT_EQ(ev.sumInstance(0, {0, 0, 3}), 1);
  EXPECT_EQ(ev.sumInstance(0, {0, 1, 3}), 10);
  EXPECT_EQ(ev.sumInstance(0, {0, 2, 3}), 100);

  EXPECT_EQ(ev.sumInstance(0, {0, 0, 1}), 111);

  EXPECT_EQ(ev.sumInstance(3, {0, 0, 1}), 444);

  // Non-zero offset
  EXPECT_EQ(ev.sumInstance(0, {1, 0, 2}), 10);
  EXPECT_EQ(ev.sumInstance(0, {1, 1, 2}), 100);
  EXPECT_EQ(ev.sumInstance(0, {1, 0, 1}), 110);

  ev.addSample(t, {1000, 2000, 3000, 4000});

  EXPECT_EQ(ev.sumInstance(0, {1, 0, 3}), 10);
  EXPECT_EQ(ev.sumInstance(0, {1, 1, 3}), 100);
  EXPECT_EQ(ev.sumInstance(0, {2, 1, 2}), 1000);
  EXPECT_EQ(ev.sumInstance(0, {2, 0, 1}), 1100);

  EXPECT_EQ(ev.sumAll({0, 0, 4}), 10);
  EXPECT_EQ(ev.sumAll({1, 0, 3}), 100);
  EXPECT_EQ(ev.sumAll({2, 1, 2}), 10000);
  EXPECT_EQ(ev.sumAll({0, 1, 2}), 11000);
  EXPECT_EQ(ev.sumAll({0, 0, 1}), 11110);
}

TEST(EventTest, Percentiles) {
  Event ev;
  ev.instanceCount = 4;
  auto t = system_clock::now();
  ev.addSample(t, {3, 2, 1, 4});
  ev.addSample(t, {30, 20, 10, 40});
  ev.addSample(t, {300, 200, 100, 400});

  PercentileList pct = {
      {10, SampleValue(0)}, {50, SampleValue(0)}, {90, SampleValue(0)}};

  ev.percentiles(pct, {0, 0, 3});
  EXPECT_EQ(pct[0].second.getInt(), 1);
  EXPECT_EQ(pct[1].second.getInt(), 3);
  EXPECT_EQ(pct[2].second.getInt(), 4);

  ev.percentiles(pct, {0, 0, 1});
  EXPECT_EQ(pct[0].second.getInt(), 111);
  EXPECT_EQ(pct[1].second.getInt(), 333);
  EXPECT_EQ(pct[2].second.getInt(), 444);
}

class MockCuptiMetrics : public CuptiMetricApi {
 public:
  MockCuptiMetrics() : CuptiMetricApi(0) {}
  MOCK_METHOD1(idFromName, CUpti_MetricID(const std::string& name));
  MOCK_METHOD1(
      events,
      std::map<CUpti_EventID, std::string>(CUpti_MetricID metric_id));
  MOCK_METHOD1(valueKind, CUpti_MetricValueKind(CUpti_MetricID metric));
  MOCK_METHOD1(
      evaluationMode,
      CUpti_MetricEvaluationMode(CUpti_MetricID metric));
  MOCK_METHOD5(
      calculate,
      SampleValue(
          CUpti_MetricID metric,
          CUpti_MetricValueKind kind,
          std::vector<CUpti_EventID>& events,
          std::vector<int64_t>& values,
          int64_t duration));
};

TEST(MetricTest, Calculate) {
  using ::testing::Return;
  MockCuptiMetrics metrics;

  // The events used for the ipc metrics: instructions and cycles
  // Pretend we have 2 SMs and 2 samples of each event
  Event instr("instructions");
  instr.instanceCount = 2;
  auto t = system_clock::now();
  instr.addSample(t, {100, 200});
  instr.addSample(t, {300, 400});

  Event cycles("cycles");
  cycles.instanceCount = 2;
  cycles.addSample(t, {1000, 1200});
  cycles.addSample(t, {1300, 1300});

  // 2 & 3 are the event ids we specified in the metric
  std::map<CUpti_EventID, Event> events;
  events[2] = std::move(instr);
  events[3] = std::move(cycles);

  // Define an ipc metric
  EXPECT_CALL(metrics, valueKind(1))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_VALUE_KIND_DOUBLE));
  Metric m(
      "ipc", 1, {2, 3}, CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE, metrics);

  // Calculate metric for first sample
  // Since evaluation mode is CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE,
  // Cupti API will be called three times: once for each SM (2) and once
  // to get the total across SMs.
  std::vector<CUpti_EventID> ids = {2, 3};
  std::vector<int64_t> vals = {100, 1000};
  EXPECT_CALL(
      metrics, calculate(1, CUPTI_METRIC_VALUE_KIND_DOUBLE, ids, vals, 1000))
      .Times(1)
      .WillOnce(Return(SampleValue(0.1)));
  vals = {200, 1200};
  EXPECT_CALL(
      metrics, calculate(1, CUPTI_METRIC_VALUE_KIND_DOUBLE, ids, vals, 1000))
      .Times(1)
      .WillOnce(Return(SampleValue(0.17)));
  vals = {300, 2200};
  EXPECT_CALL(
      metrics, calculate(1, CUPTI_METRIC_VALUE_KIND_DOUBLE, ids, vals, 1000))
      .Times(1)
      .WillOnce(Return(SampleValue(0.14)));
  auto v = m.calculate(events, nanoseconds(1000), {0, 0, 2});

  EXPECT_EQ(v.perInstance.size(), 2);
  EXPECT_EQ(v.perInstance[0].getDouble(), 0.1);
  EXPECT_EQ(v.perInstance[1].getDouble(), 0.17);
  EXPECT_EQ(v.total.getDouble(), 0.14);

  // Calculate second sample.
  // Change evaluation mode to CUPTI_METRIC_EVALUATION_MODE_AGGREGATE.
  // Now we should get only one call to the Cupti API for the total.
  EXPECT_CALL(metrics, valueKind(1))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_VALUE_KIND_DOUBLE));
  Metric m2("ipc", 1, {2, 3}, CUPTI_METRIC_EVALUATION_MODE_AGGREGATE, metrics);
  vals = {700, 2600};
  EXPECT_CALL(
      metrics, calculate(1, CUPTI_METRIC_VALUE_KIND_DOUBLE, ids, vals, 1000))
      .Times(1)
      .WillOnce(Return(SampleValue(0.27)));
  v = m2.calculate(events, nanoseconds(1000), {0, 1, 2});

  EXPECT_EQ(v.perInstance.size(), 1);
  EXPECT_EQ(v.perInstance[0].getDouble(), 0.27);
  EXPECT_EQ(v.total.getDouble(), 0.27);
}

class MockCuptiEvents : public CuptiEventApi {
 public:
  MOCK_METHOD1(
      createGroupSets,
      CUpti_EventGroupSets*(std::vector<CUpti_EventID>& ids));
  MOCK_METHOD1(destroyGroupSets, void(CUpti_EventGroupSets* sets));
  MOCK_METHOD0(setContinuousMode, bool());
  MOCK_METHOD1(enablePerInstance, void(CUpti_EventGroup eventGroup));
  MOCK_METHOD1(instanceCount, uint32_t(CUpti_EventGroup eventGroup));
  MOCK_METHOD1(enableGroupSet, void(CUpti_EventGroupSet& set));
  MOCK_METHOD1(disableGroupSet, void(CUpti_EventGroupSet& set));
  MOCK_METHOD3(
      readEvent,
      void(CUpti_EventGroup g, CUpti_EventID id, std::vector<int64_t>& vals));
  MOCK_METHOD1(eventsInGroup, std::vector<CUpti_EventID>(CUpti_EventGroup g));
  MOCK_METHOD1(eventId, CUpti_EventID(const std::string& name));
};

TEST(EventGroupSetTest, CollectSample) {
  using ::testing::_;
  using ::testing::Return;
  const CUpti_EventGroup g1{nullptr};
  const CUpti_EventGroup g2{reinterpret_cast<void*>(0x1000)};
  CUpti_EventGroup groups[] = {g1, g2};
  CUpti_EventGroupSet set;
  set.eventGroups = groups;
  set.numEventGroups = 2;

  std::map<CUpti_EventID, Event> events;
  Event instr("instructions");
  events[4] = std::move(instr);
  Event cycles("cycles");
  events[5] = std::move(cycles);
  Event branches("branches");
  events[10] = std::move(branches);

  MockCuptiEvents cupti_events;
  EXPECT_CALL(cupti_events, enablePerInstance(g1)).Times(1);
  EXPECT_CALL(cupti_events, enablePerInstance(g2)).Times(1);
  EXPECT_CALL(cupti_events, instanceCount(g1)).Times(1).WillOnce(Return(80));
  EXPECT_CALL(cupti_events, instanceCount(g2)).Times(1).WillOnce(Return(40));
  std::vector<CUpti_EventID> events_in_group1 = {4, 5};
  EXPECT_CALL(cupti_events, eventsInGroup(g1))
      .Times(1)
      .WillOnce(Return(events_in_group1));
  std::vector<CUpti_EventID> events_in_group2 = {10};
  EXPECT_CALL(cupti_events, eventsInGroup(g2))
      .Times(1)
      .WillOnce(Return(events_in_group2));
  EventGroupSet group_set(set, events, cupti_events);

  EXPECT_EQ(group_set.groupCount(), 2);
  EXPECT_EQ(events[4].instanceCount, 80);
  EXPECT_EQ(events[5].instanceCount, 80);
  EXPECT_EQ(events[10].instanceCount, 40);

  // This should not cause any Cupti API action as the group
  // set is already disabled
  group_set.setEnabled(false);

  // Activate group set - if activated twice, only the first
  // should cause cupti API to be called
  EXPECT_CALL(cupti_events, enableGroupSet(_)).Times(1);
  group_set.setEnabled(false);
  group_set.setEnabled(true);

  EXPECT_CALL(cupti_events, eventsInGroup(g1))
      .Times(1)
      .WillOnce(Return(events_in_group1));
  EXPECT_CALL(cupti_events, eventsInGroup(g2))
      .Times(1)
      .WillOnce(Return(events_in_group2));
  EXPECT_CALL(cupti_events, readEvent(g1, 4, _)).Times(1);
  EXPECT_CALL(cupti_events, readEvent(g1, 5, _)).Times(1);
  EXPECT_CALL(cupti_events, readEvent(g2, 10, _)).Times(1);
  group_set.collectSample();

  EXPECT_EQ(events[4].sampleCount(), 1);
  EXPECT_EQ(events[5].sampleCount(), 1);
  EXPECT_EQ(events[10].sampleCount(), 1);
}

class MockLogger : public SampleListener {
 public:
  MOCK_METHOD3(handleSample, void(int device, const Sample& sample, bool from_new_version));
  MOCK_METHOD1(update, void(const Config& config));
};

class EventProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto cupti_events_ptr = std::make_unique<MockCuptiEvents>();
    auto cupti_metrics_ptr = std::make_unique<MockCuptiMetrics>();
    cuptiEvents_ = cupti_events_ptr.get();
    cuptiMetrics_ = cupti_metrics_ptr.get();
    loggers_.push_back(std::make_unique<MockLogger>());
    onDemandLoggers_.push_back(std::make_unique<MockLogger>());
    profiler_ = std::make_unique<EventProfiler>(
        std::move(cupti_events_ptr),
        std::move(cupti_metrics_ptr),
        loggers_,
        onDemandLoggers_);

    for (int i = 0; i < kEventGroupCount; i++) {
      eventGroups_[i] = &eventGroups_[i];
    }
    for (int i = 0; i < kGroupSetCount; i++) {
      // Default size to 1 but can be changed by test
      groupSet_[i].numEventGroups = 1;
      // Two groups per set
      groupSet_[i].eventGroups = &eventGroups_[i * 2];
    }
    groupSets_.numSets = 1;
    groupSets_.sets = groupSet_;
  }

  MockCuptiEvents* cuptiEvents_;
  MockCuptiMetrics* cuptiMetrics_;
  std::vector<std::unique_ptr<SampleListener>> loggers_;
  std::vector<std::unique_ptr<SampleListener>> onDemandLoggers_;
  constexpr static int kEventGroupCount = 4;
  constexpr static int kGroupSetCount = 2;
  CUpti_EventGroup eventGroups_[kEventGroupCount];
  CUpti_EventGroupSet groupSet_[kGroupSetCount];
  CUpti_EventGroupSets groupSets_;
  std::unique_ptr<EventProfiler> profiler_;
};

TEST_F(EventProfilerTest, ConfigureFailure) {
  using namespace testing;

  // Default config has no counters enabled.
  // Check that profiler remains disabled.
  Config cfg;
  profiler_->configure(cfg, nullptr);

  EXPECT_FALSE(profiler_->enabled());

  // There is no event named "cycles"
  // In this case the profiler should print a warning and remain disabled
  bool parsed = cfg.parse("EVENTS = cycles");
  EXPECT_TRUE(parsed);

  // EventProfiler should handle exception thrown from createGroupSets
  // Configuration will be applied twice - once for combined base + on-demand
  // and then again falling back to base
  EXPECT_CALL(*cuptiEvents_, eventId("cycles"))
      .Times(2)
      .WillRepeatedly(Return(0));
  std::vector<CUpti_EventID> ids = {0};
  EXPECT_CALL(*cuptiEvents_, createGroupSets(ids))
      .Times(2)
      .WillRepeatedly(Throw(
          std::system_error(EINVAL, std::generic_category(), "Event ID")));
  profiler_->configure(cfg, nullptr);

  EXPECT_FALSE(profiler_->enabled());
}

TEST_F(EventProfilerTest, ConfigureBase) {
  using namespace testing;

  // Test normal path, simple base config
  Config cfg;
  bool parsed = cfg.parse("EVENTS = elapsed_cycles_sm");
  EXPECT_TRUE(parsed);

  // One valid event - expect one call to eventId and createGroupSets
  EXPECT_CALL(*cuptiEvents_, eventId("elapsed_cycles_sm"))
      .Times(1)
      .WillOnce(Return(5));
  std::vector<CUpti_EventID> ids = {5};
  EXPECT_CALL(*cuptiEvents_, createGroupSets(ids))
      .Times(1)
      .WillOnce(Return(&groupSets_));
  EXPECT_CALL(*cuptiEvents_, enablePerInstance(eventGroups_[0])).Times(1);
  EXPECT_CALL(*cuptiEvents_, instanceCount(eventGroups_[0]))
      .Times(1)
      .WillOnce(Return(80));
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[0]))
      .Times(1)
      .WillOnce(Return(ids));
  EXPECT_CALL(*cuptiEvents_, enableGroupSet(_)).Times(1);

  profiler_->configure(cfg, nullptr);

  EXPECT_TRUE(profiler_->enabled());
}

TEST_F(EventProfilerTest, ConfigureOnDemand) {
  using namespace testing;

  // Test base + on-demand config, one event and one metric
  Config cfg, on_demand_cfg;
  bool parsed = cfg.parse(R"(
    EVENTS = active_cycles
    SAMPLE_PERIOD_MSECS=500
    REPORT_PERIOD_SECS=10
    SAMPLES_PER_REPORT=5
  )");
  EXPECT_TRUE(parsed);

  parsed = on_demand_cfg.parse(R"(
    METRICS = ipc
    EVENTS_DURATION_SECS=60
    SAMPLE_PERIOD_MSECS=200
    MULTIPLEX_PERIOD_MSECS=2000
    REPORT_PERIOD_SECS=3
    SAMPLES_PER_REPORT=10
  )");
  EXPECT_TRUE(parsed);

  // One event
  EXPECT_CALL(*cuptiEvents_, eventId("active_cycles"))
      .Times(1)
      .WillOnce(Return(3));
  // One metric
  EXPECT_CALL(*cuptiMetrics_, idFromName("ipc")).Times(1).WillOnce(Return(10));
  std::map<CUpti_EventID, std::string> ipc_events;
  ipc_events[4] = "instructions";
  ipc_events[5] = "elapsed_cycles_sm";
  EXPECT_CALL(*cuptiMetrics_, events(10)).Times(1).WillOnce(Return(ipc_events));
  EXPECT_CALL(*cuptiMetrics_, evaluationMode(10))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE));
  EXPECT_CALL(*cuptiMetrics_, valueKind(10))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_VALUE_KIND_DOUBLE));
  std::vector<CUpti_EventID> ids = {3, 4, 5};
  groupSet_[0].numEventGroups = 2;
  groupSets_.numSets = 2;
  EXPECT_CALL(*cuptiEvents_, createGroupSets(ids))
      .Times(1)
      .WillOnce(Return(&groupSets_));
  // Specified CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE per instance above
  // So check that it's enabled
  EXPECT_CALL(*cuptiEvents_, enablePerInstance(eventGroups_[0])).Times(1);
  EXPECT_CALL(*cuptiEvents_, enablePerInstance(eventGroups_[1])).Times(1);
  EXPECT_CALL(*cuptiEvents_, enablePerInstance(eventGroups_[2])).Times(1);
  std::vector<CUpti_EventID> ids_g1{3}, ids_g2{4}, ids_g3{5};
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[0]))
      .Times(1)
      .WillOnce(Return(ids_g1));
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[1]))
      .Times(1)
      .WillOnce(Return(ids_g2));
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[2]))
      .Times(1)
      .WillOnce(Return(ids_g3));
  EXPECT_CALL(*cuptiEvents_, enableGroupSet(_)).Times(1);

  profiler_->configure(cfg, &on_demand_cfg);

  EXPECT_TRUE(profiler_->enabled());
  EXPECT_EQ(profiler_->samplePeriod().count(), 250);
  EXPECT_EQ(profiler_->multiplexPeriod().count(), 1000);
  EXPECT_EQ(profiler_->reportPeriod().count(), 10000);
  EXPECT_EQ(profiler_->onDemandReportPeriod().count(), 4000);
}

TEST_F(EventProfilerTest, ReportSample) {
  using namespace testing;

  // Test base + on-demand config, one event and one metric
  Config cfg, on_demand_cfg;
  bool parsed = cfg.parse("EVENTS = active_cycles");
  EXPECT_TRUE(parsed);

  parsed = on_demand_cfg.parse(R"(
    METRICS = ipc
    EVENTS_DURATION_SECS=60
  )");
  EXPECT_TRUE(parsed);

  // One event
  EXPECT_CALL(*cuptiEvents_, eventId("active_cycles"))
      .Times(1)
      .WillOnce(Return(3));
  // One metric
  EXPECT_CALL(*cuptiMetrics_, idFromName("ipc")).Times(1).WillOnce(Return(10));
  std::map<CUpti_EventID, std::string> ipc_events;
  ipc_events[4] = "instructions";
  ipc_events[5] = "elapsed_cycles_sm";
  EXPECT_CALL(*cuptiMetrics_, events(10)).Times(1).WillOnce(Return(ipc_events));
  EXPECT_CALL(*cuptiMetrics_, evaluationMode(10))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE));
  EXPECT_CALL(*cuptiMetrics_, valueKind(10))
      .Times(1)
      .WillOnce(Return(CUPTI_METRIC_VALUE_KIND_DOUBLE));
  std::vector<CUpti_EventID> ids = {3, 4, 5};
  groupSet_[0].numEventGroups = 2;
  groupSets_.numSets = 2;
  EXPECT_CALL(*cuptiEvents_, createGroupSets(ids))
      .Times(1)
      .WillOnce(Return(&groupSets_));
  EXPECT_CALL(*cuptiEvents_, instanceCount(_))
      .Times(3)
      .WillRepeatedly(Return(4));
  std::vector<CUpti_EventID> ids_g1{3}, ids_g2{4}, ids_g3{5};
  // These will be called by collectSample() as well, which is called twice
  // per group set
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[0]))
      .Times(3)
      .WillRepeatedly(Return(ids_g1));
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[1]))
      .Times(3)
      .WillRepeatedly(Return(ids_g2));
  EXPECT_CALL(*cuptiEvents_, eventsInGroup(eventGroups_[2]))
      .Times(3)
      .WillRepeatedly(Return(ids_g3));
  EXPECT_CALL(*cuptiEvents_, enableGroupSet(_)).Times(1);

  profiler_->configure(cfg, &on_demand_cfg);

  EXPECT_TRUE(profiler_->enabled());

  EXPECT_CALL(*cuptiEvents_, readEvent(_, _, _))
      .Times(6)
      .WillRepeatedly(Invoke(
          [](CUpti_EventGroup g, CUpti_EventID id, std::vector<int64_t>& vals) {
            vals = {1, 2, 3, 4};
          }));

  // Need to collect four times - twice for each group set
  profiler_->collectSample();
  profiler_->collectSample();
  EXPECT_CALL(*cuptiEvents_, disableGroupSet(_)).Times(1);
  EXPECT_CALL(*cuptiEvents_, enableGroupSet(_)).Times(1);
  profiler_->enableNextCounterSet();
  profiler_->collectSample();
  profiler_->collectSample();

  std::vector<CUpti_EventID> ipc_ids = {4, 5};
  // Called once for each instance (4) and once for the total.
  // x2 since we recompute per logger.
  EXPECT_CALL(
      *cuptiMetrics_,
      calculate(10, CUPTI_METRIC_VALUE_KIND_DOUBLE, ipc_ids, _, 2000000000))
      .Times(10)
      .WillRepeatedly(Return(SampleValue(0.3)));
  auto& logger = dynamic_cast<MockLogger&>(*loggers_[0]);
  EXPECT_CALL(logger, handleSample(0, _, _))
      .Times(1)
      .WillOnce(Invoke([](int device, const Sample& sample, bool from_new_version) {
        // Sample will include all stats - logger must pick the
        // ones it wants.
        EXPECT_EQ(sample.stats.size(), 4);
        EXPECT_EQ(sample.stats[0].name, "active_cycles");
        EXPECT_EQ(sample.stats[1].name, "instructions");
        EXPECT_EQ(sample.stats[2].name, "elapsed_cycles_sm");
        EXPECT_EQ(sample.stats[3].name, "ipc");
        // 2 samples, each with values {1, 2, 3, 4}
        // i.e. {2, 4, 6, 8} total
        EXPECT_EQ(sample.stats[0].total.getInt(), 20);
        EXPECT_EQ(sample.stats[0].percentileValues[0].second.getInt(), 2);
        EXPECT_EQ(sample.stats[0].percentileValues.back().second.getInt(), 8);
        // ipc is always 0.3 from mocked calculate function above
        EXPECT_EQ(sample.stats[3].total.getDouble(), 0.3);
        EXPECT_EQ(sample.stats[3].percentileValues[0].second.getDouble(), 0.3);
        EXPECT_EQ(
            sample.stats[3].percentileValues.back().second.getDouble(), 0.3);
      }));
  profiler_->reportSamples();

  auto& on_demand_logger = dynamic_cast<MockLogger&>(*onDemandLoggers_[0]);
  EXPECT_CALL(on_demand_logger, handleSample(0, _, _)).Times(1);
  profiler_->reportOnDemandSamples();

  EXPECT_CALL(*cuptiEvents_, disableGroupSet(_)).Times(1);
}
