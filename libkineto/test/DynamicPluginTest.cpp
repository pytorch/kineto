/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "include/DynamicPluginInterface.h"
#include "src/dynamic_plugin/PluginProfiler.h"
#include "src/output_membuf.h"

using namespace KINETO_NAMESPACE;

namespace {

// Mock Plugin Handle - simple struct to track state
struct MockPluginHandle {
  bool created = false;
  bool active = false;
  std::string name = "MockPlugin";
};

// Mock implementation of the plugin interface
class MockPlugin {
public:
  static std::vector<std::unique_ptr<MockPluginHandle>> handles;

  static int getHandleCount() { return handles.size(); }

  // Reset state for clean tests
  static void reset() { handles.clear(); }

  // Mock profilerCreate implementation
  static int profilerCreate(KinetoPlugin_ProfilerCreate_Params *params) {
    handles.push_back(std::make_unique<MockPluginHandle>());
    MockPluginHandle *handle = handles.back().get();
    handle->created = true;
    params->pProfilerHandle =
        reinterpret_cast<KinetoPlugin_ProfilerHandle *>(handle);
    return 0;
  }

  // Mock profilerDestroy implementation
  static int profilerDestroy(KinetoPlugin_ProfilerDestroy_Params *params) {
    MockPluginHandle *handle =
        reinterpret_cast<MockPluginHandle *>(params->pProfilerHandle);

    // Find and remove from our tracking
    for (auto it = handles.begin(); it != handles.end(); ++it) {
      if (it->get() == handle) {
        handles.erase(it);
        break;
      }
    }
    return 0;
  }

  // Mock profilerQuery implementation
  static int profilerQuery(KinetoPlugin_ProfilerQuery_Params *params) {
    strncpy(params->pProfilerName, "MockPlugin", params->profilerNameMaxLen);
    params->pProfilerName[params->profilerNameMaxLen] = '\0';
    return 0;
  }

  // Mock profilerStart implementation
  static int profilerStart(KinetoPlugin_ProfilerStart_Params *params) {
    MockPluginHandle *handle =
        reinterpret_cast<MockPluginHandle *>(params->pProfilerHandle);
    handle->active = true;
    return 0;
  }

  // Mock profilerStop implementation
  static int profilerStop(KinetoPlugin_ProfilerStop_Params *params) {
    MockPluginHandle *handle =
        reinterpret_cast<MockPluginHandle *>(params->pProfilerHandle);
    handle->active = false;
    return 0;
  }

  // Mock profilerProcessEvents implementation
  static int
  profilerProcessEvents(KinetoPlugin_ProfilerProcessEvents_Params *params) {
    const KinetoPlugin_TraceBuilder *traceBuilder = params->pTraceBuilder;

    // Create sample events of different types
    int64_t baseTime = 1000000000; // 1 second in nanoseconds

    // 1. Runtime activity (CUDA runtime API call)
    KinetoPlugin_ProfileEvent runtimeEvent{
        .unpaddedStructSize = KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE,
        .eventType = KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME,
        .startTimeUtcNs = baseTime,
        .endTimeUtcNs = baseTime + 5000,
        .eventId = 1,
        .deviceId = 0,
        .resourceId = 123};
    traceBuilder->addEvent(traceBuilder->pTraceBuilderHandle, &runtimeEvent);
    traceBuilder->setLastEventName(traceBuilder->pTraceBuilderHandle,
                                   "cudaLaunchKernel");

    // 2. Driver activity (CUDA driver API call)
    KinetoPlugin_ProfileEvent driverEvent{
        .unpaddedStructSize = KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE,
        .eventType = KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_DRIVER,
        .startTimeUtcNs = baseTime + 10000,
        .endTimeUtcNs = baseTime + 15000,
        .eventId = 2,
        .deviceId = 0,
        .resourceId = 124};
    traceBuilder->addEvent(traceBuilder->pTraceBuilderHandle, &driverEvent);
    traceBuilder->setLastEventName(traceBuilder->pTraceBuilderHandle,
                                   "cuLaunchKernel");

    // 3. Kernel activity (GPU kernel execution)
    KinetoPlugin_ProfileEvent kernelEvent{
        .unpaddedStructSize = KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE,
        .eventType = KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL,
        .startTimeUtcNs = baseTime + 20000,
        .endTimeUtcNs = baseTime + 50000,
        .eventId = 3,
        .deviceId = 0,
        .resourceId = 1};
    traceBuilder->addEvent(traceBuilder->pTraceBuilderHandle, &kernelEvent);
    traceBuilder->setLastEventName(traceBuilder->pTraceBuilderHandle,
                                   "test_kernel");

    // 4. Memcpy activity (GPU memory copy)
    KinetoPlugin_ProfileEvent memcpyEvent{
        .unpaddedStructSize = KINETO_PLUGIN_PROFILE_EVENT_UNPADDED_STRUCT_SIZE,
        .eventType = KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMCPY,
        .startTimeUtcNs = baseTime + 60000,
        .endTimeUtcNs = baseTime + 70000,
        .eventId = 4,
        .deviceId = 0,
        .resourceId = 2};
    traceBuilder->addEvent(traceBuilder->pTraceBuilderHandle, &memcpyEvent);
    traceBuilder->setLastEventName(traceBuilder->pTraceBuilderHandle,
                                   "cudaMemcpyHtoD");

    return 0;
  }

  // Get the mock profiler interface
  static KinetoPlugin_ProfilerInterface getInterface() {
    return KinetoPlugin_ProfilerInterface{
        .unpaddedStructSize =
            KINETO_PLUGIN_PROFILER_INTERFACE_UNPADDED_STRUCT_SIZE,
        .profilerCreate = profilerCreate,
        .profilerDestroy = profilerDestroy,
        .profilerQuery = profilerQuery,
        .profilerStart = profilerStart,
        .profilerStop = profilerStop,
        .profilerProcessEvents = profilerProcessEvents};
  }
};

// Static member definitions
std::vector<std::unique_ptr<MockPluginHandle>> MockPlugin::handles;

} // anonymous namespace

class DynamicPluginTest : public ::testing::Test {
protected:
  void SetUp() override { MockPlugin::reset(); }

  void TearDown() override { MockPlugin::reset(); }
};

// Test complete plugin lifecycle through PluginProfiler
TEST_F(DynamicPluginTest, PluginProfilerLifecycle) {
  auto mockInterface = MockPlugin::getInterface();

  // Create a PluginProfiler instance with our mock
  PluginProfiler pluginProfiler(mockInterface);

  // Test that the name is correctly retrieved
  EXPECT_EQ(pluginProfiler.name(), "MockPlugin");

  // Test that available activities are returned
  auto activities = pluginProfiler.availableActivities();
  EXPECT_FALSE(
      activities
          .empty()); // Should have at least the default CUDA_PROFILER_RANGE

  // Create a profiler session which will test handle creation
  EXPECT_EQ(MockPlugin::getHandleCount(), 0);
  auto session = pluginProfiler.configure(activities, Config{});
  EXPECT_NE(session, nullptr);
  EXPECT_EQ(MockPlugin::getHandleCount(), 1);

  // Verify handle was created and is in correct state
  MockPluginHandle *handle = MockPlugin::handles[0].get();
  EXPECT_TRUE(handle->created);
  EXPECT_FALSE(handle->active);

  // Test start profiling
  session->start();
  EXPECT_TRUE(handle->active);

  // Test stop profiling
  session->stop();
  EXPECT_FALSE(handle->active);

  // Test session destruction (handle cleanup)
  session.reset();
  EXPECT_EQ(MockPlugin::getHandleCount(), 0);
}

// Test event builder functionality with processEvents
TEST_F(DynamicPluginTest, EventBuilderProcessing) {
  auto mockInterface = MockPlugin::getInterface();

  // Create a PluginProfiler instance
  PluginProfiler pluginProfiler(mockInterface);

  // Create and configure a session
  auto activities = pluginProfiler.availableActivities();
  auto session = pluginProfiler.configure(activities, Config{});
  EXPECT_NE(session, nullptr);

  // Start profiling
  session->start();

  // Stop profiling
  session->stop();

  // Process events - this will call our mock profilerProcessEvents
  // which creates sample events using the trace builder
  MemoryTraceLogger logger(Config{});
  session->processTrace(logger);

  // Get the trace buffer to verify events were created
  auto traceBuffer = session->getTraceBuffer();
  EXPECT_NE(traceBuffer, nullptr);

  // Verify we have the expected number of activities (4 events created in mock)
  EXPECT_EQ(traceBuffer->activities.size(), 4);

  // Verify the different activity types were created correctly
  auto &activities_vec = traceBuffer->activities;

  // Check runtime activity (first event)
  EXPECT_EQ(activities_vec[0]->type(), ActivityType::CUDA_RUNTIME);
  EXPECT_EQ(activities_vec[0]->activityName, "cudaLaunchKernel");
  EXPECT_EQ(activities_vec[0]->startTime, 1000000000);
  EXPECT_EQ(activities_vec[0]->endTime, 1000005000);
  EXPECT_EQ(activities_vec[0]->id, 1);

  // Check driver activity (second event)
  EXPECT_EQ(activities_vec[1]->type(), ActivityType::CUDA_DRIVER);
  EXPECT_EQ(activities_vec[1]->activityName, "cuLaunchKernel");
  EXPECT_EQ(activities_vec[1]->startTime, 1000010000);
  EXPECT_EQ(activities_vec[1]->endTime, 1000015000);
  EXPECT_EQ(activities_vec[1]->id, 2);

  // Check kernel activity (third event)
  EXPECT_EQ(activities_vec[2]->type(), ActivityType::CONCURRENT_KERNEL);
  EXPECT_EQ(activities_vec[2]->activityName, "test_kernel");
  EXPECT_EQ(activities_vec[2]->startTime, 1000020000);
  EXPECT_EQ(activities_vec[2]->endTime, 1000050000);
  EXPECT_EQ(activities_vec[2]->id, 3);

  // Check memcpy activity (fourth event)
  EXPECT_EQ(activities_vec[3]->type(), ActivityType::GPU_MEMCPY);
  EXPECT_EQ(activities_vec[3]->activityName, "cudaMemcpyHtoD");
  EXPECT_EQ(activities_vec[3]->startTime, 1000060000);
  EXPECT_EQ(activities_vec[3]->endTime, 1000070000);
  EXPECT_EQ(activities_vec[3]->id, 4);
}
