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

    // Set supported activity types - simulate a plugin that supports CUDA
    // activities
    if (params->pSupportedActivityTypes &&
        params->supportedActivityTypesMaxLen > 3) {
      params->pSupportedActivityTypes[0] =
          KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_RUNTIME;
      params->pSupportedActivityTypes[1] =
          KINETO_PLUGIN_PROFILE_EVENT_TYPE_CUDA_DRIVER;
      params->pSupportedActivityTypes[2] =
          KINETO_PLUGIN_PROFILE_EVENT_TYPE_CONCURRENT_KERNEL;
      params->pSupportedActivityTypes[3] =
          KINETO_PLUGIN_PROFILE_EVENT_TYPE_GPU_MEMCPY;

      // Fill remaining slots with invalid type
      for (size_t i = 4; i < params->supportedActivityTypesMaxLen; i++) {
        params->pSupportedActivityTypes[i] =
            KINETO_PLUGIN_PROFILE_EVENT_TYPE_INVALID;
      }
    }

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

// Test configure() with activity types that intersect with plugin support
// This tests the "ANY" logic - profiler should be enabled if it supports ANY of
// the requested types
TEST_F(DynamicPluginTest, ConfigureActivityTypes) {
  auto mockInterface = MockPlugin::getInterface();
  PluginProfiler pluginProfiler(mockInterface);

  // Test that the plugin correctly reports supported activities
  auto supportedActivities = pluginProfiler.availableActivities();
  EXPECT_FALSE(supportedActivities.empty());

  // Verify specific supported activities (based on our mock implementation)
  EXPECT_TRUE(supportedActivities.find(ActivityType::CUDA_RUNTIME) !=
              supportedActivities.end());
  EXPECT_TRUE(supportedActivities.find(ActivityType::CUDA_DRIVER) !=
              supportedActivities.end());
  EXPECT_TRUE(supportedActivities.find(ActivityType::CONCURRENT_KERNEL) !=
              supportedActivities.end());
  EXPECT_TRUE(supportedActivities.find(ActivityType::GPU_MEMCPY) !=
              supportedActivities.end());

  // Test case 1: Request activities that are all supported
  std::set<ActivityType> allSupportedTypes = {
      ActivityType::CUDA_RUNTIME, ActivityType::CUDA_DRIVER,
      ActivityType::CONCURRENT_KERNEL, ActivityType::GPU_MEMCPY};

  auto session1 = pluginProfiler.configure(allSupportedTypes, Config{});
  EXPECT_NE(session1, nullptr)
      << "Should succeed when all requested types are supported";

  // Test case 2: Request activities where some are supported (intersection
  // exists)
  std::set<ActivityType> partialSupportedTypes = {
      ActivityType::CUDA_RUNTIME,      // Supported
      ActivityType::CPU_OP,            // Not supported
      ActivityType::CONCURRENT_KERNEL, // Supported
      ActivityType::USER_ANNOTATION    // Not supported
  };

  auto session2 = pluginProfiler.configure(partialSupportedTypes, Config{});
  EXPECT_NE(session2, nullptr)
      << "Should succeed when ANY requested types are supported";

  // Test configure() with activity types that don't intersect with plugin
  // support This tests the failure case - profiler should NOT be enabled if it
  // supports NONE of the requested types

  // Test case 3: Request activities that are completely unsupported
  std::set<ActivityType> unsupportedTypes = {
      ActivityType::CPU_OP, ActivityType::USER_ANNOTATION,
      ActivityType::PYTHON_FUNCTION, ActivityType::OVERHEAD};

  auto session3 = pluginProfiler.configure(unsupportedTypes, Config{});
  EXPECT_EQ(session3, nullptr)
      << "Should fail when NO requested types are supported";

  // Test case 4: Request single unsupported activity type
  std::set<ActivityType> singleUnsupportedType = {ActivityType::CPU_OP};

  auto session4 = pluginProfiler.configure(singleUnsupportedType, Config{});
  EXPECT_EQ(session4, nullptr)
      << "Should fail with single unsupported activity type";

  // Test case 5: Empty activity types set
  std::set<ActivityType> emptyTypes = {};

  auto session5 = pluginProfiler.configure(emptyTypes, Config{});
  EXPECT_EQ(session5, nullptr) << "Should fail with empty activity types set";
}
