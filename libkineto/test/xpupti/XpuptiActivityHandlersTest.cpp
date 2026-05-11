/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/plugin/xpupti/XpuptiActivityApi.h"
#include "src/plugin/xpupti/XpuptiActivityProfilerSession.h"
#include "src/ActivityBuffers.h"
#include "include/output_base.h"

#include "src/plugin/xpupti/XpuptiProfilerMacros.h"

#include <gtest/gtest.h>

namespace KN = KINETO_NAMESPACE;
using namespace libkineto;

// Mock XpuptiActivityApi that delivers hand-crafted PTI records
// through the virtual processActivities without needing PTI runtime.
class MockXpuptiActivityApi : public KN::XpuptiActivityApi {
 public:
  std::vector<const pti_view_record_base*> records;

  std::unique_ptr<KN::XpuptiActivityBufferMap> activityBuffers() override {
    // Return a non-null map so processTrace enters the processing path.
    return std::make_unique<KN::XpuptiActivityBufferMap>();
  }

  const std::pair<int, int> processActivities(
      KN::XpuptiActivityBufferMap&,
      std::function<void(const pti_view_record_base*)> handler) override {
    for (auto* record : records) {
      handler(record);
    }
    return {static_cast<int>(records.size()), 0};
  }
};

// Minimal ActivityLogger that captures logged GenericTraceActivity objects.
class MockActivityLogger : public ActivityLogger {
 public:
  std::vector<const GenericTraceActivity*> logged_activities;

  void handleDeviceInfo(const DeviceInfo&, uint64_t) override {}
  void handleResourceInfo(const ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(const OverheadInfo&, int64_t) override {}
  void handleTraceSpan(const TraceSpan&) override {}

  void handleActivity(const ITraceActivity&) override {}

  void handleGenericActivity(const GenericTraceActivity& activity) override {
    logged_activities.push_back(&activity);
  }

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}

  void finalizeMemoryTrace(const std::string&, const Config&) override {}

  void finalizeTrace(
      const Config&,
      std::unique_ptr<KINETO_NAMESPACE::ActivityBuffers>,
      int64_t,
      std::unordered_map<std::string, std::vector<std::string>>&) override {}
};

class XpuptiActivityHandlersTest : public ::testing::Test {
 protected:
  MockXpuptiActivityApi mockApi_;
  MockActivityLogger logger_;

  // Processes all records in mockApi_ through the handler pipeline
  // and returns the resulting trace buffer.
  std::unique_ptr<CpuTraceBuffer> processAndGetTrace(
      int64_t windowStart = 0,
      int64_t windowEnd = 1000) {
    Config config;
    std::set<ActivityType> activity_types = {ActivityType::COLLECTIVE_COMM, ActivityType::XPU_SYNC};
    auto session = std::make_unique<KN::XpuptiActivityProfilerSession>(
        mockApi_, "__test_profiler__", config, activity_types);
    session->processTrace(
        logger_,
        [](int64_t) -> const ITraceActivity* { return nullptr; },
        windowStart,
        windowEnd);
    return session->getTraceBuffer();
  }
};

// --- Communication Activity Tests ---

#if PTI_VERSION_AT_LEAST(0, 17)
TEST_F(XpuptiActivityHandlersTest, CommunicationActivityHasXcclPrefix) {
  pti_view_record_comms comms_record{};
  comms_record._view_kind._view_kind = PTI_VIEW_COMMUNICATION;
  comms_record._name = "allreduce";
  comms_record._start_timestamp = 100;
  comms_record._end_timestamp = 200;
  comms_record._process_id = 1;
  comms_record._thread_id = 42;
  comms_record._communicator_id = 7;

  mockApi_.records.push_back(
      reinterpret_cast<const pti_view_record_base*>(&comms_record));

  auto traceBuffer = processAndGetTrace();
  ASSERT_EQ(traceBuffer->activities.size(), 1);

  auto& activity = *traceBuffer->activities[0];
  EXPECT_EQ(activity.name(), "xccl::allreduce");
  EXPECT_EQ(activity.type(), ActivityType::COLLECTIVE_COMM);
}

TEST_F(XpuptiActivityHandlersTest, CommunicationActivityFields) {
  pti_view_record_comms comms_record{};
  comms_record._view_kind._view_kind = PTI_VIEW_COMMUNICATION;
  comms_record._name = "broadcast";
  comms_record._start_timestamp = 300;
  comms_record._end_timestamp = 500;
  comms_record._process_id = 10;
  comms_record._thread_id = 77;
  comms_record._communicator_id = 99;

  mockApi_.records.push_back(
      reinterpret_cast<const pti_view_record_base*>(&comms_record));

  auto traceBuffer = processAndGetTrace();
  ASSERT_EQ(traceBuffer->activities.size(), 1);

  auto& activity = *traceBuffer->activities[0];
  EXPECT_EQ(activity.timestamp(), 300);
  EXPECT_EQ(activity.duration(), 200);
  EXPECT_EQ(activity.deviceId(), 10);
  EXPECT_EQ(activity.resourceId(), 77);
  EXPECT_EQ(activity.getThreadId(), 77);
  EXPECT_EQ(activity.getMetadataValue("Communicator_id"), "99");
}

TEST_F(XpuptiActivityHandlersTest, CommunicationActivityOutOfRange) {
  pti_view_record_comms comms_record{};
  comms_record._view_kind._view_kind = PTI_VIEW_COMMUNICATION;
  comms_record._name = "allgather";
  comms_record._start_timestamp = 2000;
  comms_record._end_timestamp = 3000;
  comms_record._process_id = 1;
  comms_record._thread_id = 1;
  comms_record._communicator_id = 1;

  mockApi_.records.push_back(
      reinterpret_cast<const pti_view_record_base*>(&comms_record));

  auto traceBuffer = processAndGetTrace(100, 500);
  EXPECT_EQ(traceBuffer->activities.size(), 0);
}
#endif // PTI_VERSION_AT_LEAST(0, 17)
