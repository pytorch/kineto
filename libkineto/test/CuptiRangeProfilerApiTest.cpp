/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <array>
#include <set>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "include/Config.h"
#include "include/libkineto.h"
#include "src/CuptiRangeProfilerApi.h"
#include "src/KernelRegistry.h"

#include "CuptiRangeProfilerTestUtil.h"
#include "src/Logger.h"

using namespace KINETO_NAMESPACE;

#if HAS_CUPTI_RANGE_PROFILER

std::unordered_map<int, CuptiProfilerResult>&
MockCuptiRBProfilerSession::getResults() {
  static std::unordered_map<int, CuptiProfilerResult> results;
  return results;
}

MockCuptiRBProfilerSessionFactory mfactory{};

TEST(CuptiRangeProfilerApiTest, contextTracking) {
  std::vector<std::string> log_modules({"CuptiRangeProfilerApi.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  std::array<int64_t, 3> data;
  std::array<CUcontext, 3> contexts;
  for (int i = 0; i < data.size(); i++) {
    contexts[i] = reinterpret_cast<CUcontext>(&data[i]);
  }

  // simulate creating contexts, this calls the trackCudaContexts
  // function that would otherwise be called via a callback
  uint32_t dev = 0;
  for (auto ctx : contexts) {
    simulateCudaContextCreate(ctx, dev++);
  }

  EXPECT_EQ(
      CuptiRBProfilerSession::getActiveDevices(),
      std::set<uint32_t>({0, 1, 2}));

  simulateCudaContextDestroy(contexts[1], 1);

  EXPECT_EQ(
      CuptiRBProfilerSession::getActiveDevices(), std::set<uint32_t>({0, 2}));

  simulateCudaContextDestroy(contexts[0], 0);
  simulateCudaContextDestroy(contexts[2], 2);

  EXPECT_TRUE(CuptiRBProfilerSession::getActiveDevices().empty());
}

TEST(CuptiRangeProfilerApiTest, asyncLaunchUserRange) {
  std::vector<std::string> log_modules({"CuptiRangeProfilerApi.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  // this is bad but the pointer is never accessed
  CUcontext ctx0 = reinterpret_cast<CUcontext>(10);
  simulateCudaContextCreate(ctx0, 0 /*device_id*/);

  CuptiRangeProfilerOptions opts{
      .metricNames = {"metricNames"},
      .deviceId = 0,
      .maxRanges = 1,
      .numNestingLevels = 1,
      .cuContext = ctx0};

  std::unique_ptr<CuptiRBProfilerSession> session_ = mfactory.make(opts);
  auto session = mfactory.asDerived(session_.get());

  session->asyncStartAndEnable(CUPTI_UserRange, CUPTI_UserReplay);

  simulateKernelLaunch(ctx0, "hello", 1);
  simulateKernelLaunch(ctx0, "foo", 2);
  simulateKernelLaunch(ctx0, "bar", 3);

  session->asyncDisableAndStop();
  // stop happens after next kernel is run
  simulateKernelLaunch(ctx0, "bar", 4);
  simulateCudaContextDestroy(ctx0, 0 /*device_id*/);

  EXPECT_EQ(session->passes_ended, 1);
  EXPECT_EQ(session->ranges_ended, 1);
  EXPECT_TRUE(session->enabled);
}

TEST(CuptiRangeProfilerApiTest, asyncLaunchAutoRange) {
  std::vector<std::string> log_modules({"CuptiRangeProfilerApi.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  // this is bad but the pointer is never accessed
  CUcontext ctx0 = reinterpret_cast<CUcontext>(10);
  CUcontext ctx1 = reinterpret_cast<CUcontext>(11);

  simulateCudaContextCreate(ctx0, 0 /*device_id*/);

  CuptiRangeProfilerOptions opts{
      .metricNames = {"metricNames"},
      .deviceId = 0,
      .maxRanges = 1,
      .numNestingLevels = 1,
      .cuContext = ctx0,
      .has_gpu_activities_enabled_ = true};

  std::unique_ptr<CuptiRBProfilerSession> session_ = mfactory.make(opts);
  auto session = mfactory.asDerived(session_.get());

  session->asyncStartAndEnable(CUPTI_AutoRange, CUPTI_KernelReplay);

  simulateKernelLaunch(ctx0, "hello", 1);
  simulateKernelLaunch(ctx0, "foo", 2);
  simulateKernelLaunch(ctx1, "kernel_on_different_device", 3);
  simulateKernelLaunch(ctx0, "bar", 77);

  session->asyncDisableAndStop();
  // stop happens after next kernel is run
  simulateKernelLaunch(ctx0, "bar", 5);

  auto V0 = KernelRegistry::singleton()->getKernelInfo(0, 0);
  auto V1 = KernelRegistry::singleton()->getKernelInfo(0, 1);
  auto V2 = KernelRegistry::singleton()->getKernelInfo(0, 2);
  auto V3 = KernelRegistry::singleton()->getKernelInfo(0, 3);
  EXPECT_TRUE(V0.has_value());
  EXPECT_TRUE(V1.has_value());
  EXPECT_TRUE(V2.has_value());
  EXPECT_FALSE(V3.has_value());

  EXPECT_EQ(V0.value().first, "hello");
  EXPECT_EQ(V1.value().first, "foo");
  EXPECT_EQ(V2.value().first, "bar");

  EXPECT_EQ(V0.value().second, 1);
  EXPECT_EQ(V1.value().second, 2);
  EXPECT_EQ(V2.value().second, 77);

  simulateCudaContextDestroy(ctx0, 0 /*device_id*/);

  EXPECT_EQ(session->passes_ended, 0);
  EXPECT_EQ(session->ranges_ended, 0);
  EXPECT_TRUE(session->enabled);
}

#endif // HAS_CUPTI_RANGE_PROFILER
