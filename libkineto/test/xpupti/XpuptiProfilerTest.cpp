/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiTestUtilities.h"

#include "include/libkineto.h"

#include <gtest/gtest.h>

#include <unordered_set>

namespace KN = KINETO_NAMESPACE;

TEST(XpuptiProfilerTest, XpuDriverEvents) {
  KN::Config cfg;

  std::vector<std::string_view> metrics;

  std::set<KN::ActivityType> activities{
      KN::ActivityType::XPU_RUNTIME,
      KN::ActivityType::XPU_DRIVER,
  };

  std::vector<std::string_view> expectedActivities = {
      "urEnqueueMemBufferWrite",
      "urEnqueueMemBufferWrite",
      "urEnqueueMemBufferWrite",
      "urEnqueueKernelLaunch",
      "urEnqueueMemBufferRead",
  };

  std::vector<std::string_view> expectedTypes = {
      "xpu_runtime",
      "xpu_runtime",
      "xpu_runtime",
      "xpu_runtime",
      "xpu_runtime"};

  constexpr unsigned repeatCount = 1;
  RunProfilerTest(
      metrics,
      activities,
      cfg,
      repeatCount,
      std::move(expectedActivities),
      std::move(expectedTypes));
}

TEST(XpuptiProfilerTest, OverheadActivity) {
  KN::Config cfg;

  std::vector<std::string_view> metrics;

  // Profile a normal kernel workload alongside the OVERHEAD activity so that
  // PTI has collection work to attribute overhead to.
  std::set<KN::ActivityType> activities{
      KN::ActivityType::CONCURRENT_KERNEL,
      KN::ActivityType::OVERHEAD,
  };

  std::vector<std::string_view> expectedActivities = {
      "Run(sycl::_V1::queue, ...)"};
  std::vector<std::string_view> expectedTypes = {"kernel"};

  constexpr unsigned repeatCount = 1;
  auto [pSession, pBuffer] = RunProfilerTest(
      metrics,
      activities,
      cfg,
      repeatCount,
      std::move(expectedActivities),
      std::move(expectedTypes));

  // Verify the shape of every overhead record the run produced.
  static const std::unordered_set<std::string> kExpectedNames{
      "Unknown", "Resource", "Buffer Flush", "Driver", "Instrumentation"};
  for (auto&& pActivity : pBuffer->activities) {
    if (pActivity->type() != KN::ActivityType::OVERHEAD) {
      continue;
    }

    // Overhead is host-side, not attributed to a device.
    EXPECT_EQ(pActivity->deviceId(), -1);

    const std::string name = pActivity->name();
    EXPECT_TRUE(kExpectedNames.count(name) == 1)
        << "unexpected overhead name: " << name;

    const auto metadata = pActivity->metadataJson();
    EXPECT_NE(metadata.find("overhead cost"), std::string::npos)
        << "metadata = " << metadata;
    EXPECT_NE(metadata.find("overhead count"), std::string::npos)
        << "metadata = " << metadata;
    EXPECT_NE(metadata.find("overhead occupancy"), std::string::npos)
        << "metadata = " << metadata;
  }
}

TEST(XpuptiProfilerTest, TestEvents) {
  KN::Config cfg;

  std::vector<std::string_view> metrics;

  std::set<KN::ActivityType> activities{
      KN::ActivityType::GPU_MEMCPY,
      KN::ActivityType::GPU_MEMSET,
      KN::ActivityType::CONCURRENT_KERNEL,
      KN::ActivityType::EXTERNAL_CORRELATION};

  std::vector<std::string_view> expectedActivities = {
      "Memcpy M2D",
      "Memcpy M2D",
      "Memcpy M2D",
      "Run(sycl::_V1::queue, ...)",
      "Memcpy D2M"};

  std::vector<std::string_view> expectedTypes = {
      "gpu_memcpy", "gpu_memcpy", "gpu_memcpy", "kernel", "gpu_memcpy"};

  constexpr unsigned repeatCount = 1;
  auto [pSession, pBuffer] = RunProfilerTest(
      metrics,
      activities,
      cfg,
      repeatCount,
      std::move(expectedActivities),
      std::move(expectedTypes));

  static bool isVerbose = IsEnvVerbose();

  auto resourceInfos = pSession->getResourceInfos();
  if (isVerbose) {
    for (auto&& ri : resourceInfos) {
#define PRINT(R) std::cout << #R " = " << ri.R << std::endl;
      PRINT(id)
      PRINT(sortIndex)
      PRINT(deviceId)
      PRINT(name)
#undef PRINT
    }
  }

  std::vector<unsigned> irUseCount(resourceInfos.size(), 0);
  for (auto&& pActivity : pBuffer->activities) {
    bool found = false;
    auto resourceId = pActivity->resourceId();
    auto deviceId = pActivity->deviceId();
    for (unsigned i = 0; i < resourceInfos.size(); ++i) {
      const auto& ri = resourceInfos[i];
      if ((ri.id == resourceId) && (ri.deviceId == deviceId)) {
        ++irUseCount[i];
        found = true;
        break;
      }
    }

    EXPECT_TRUE(found) << "resourceInfo for deviceId = " << deviceId
                       << ", resourceId=" << resourceId << " not found.";
  }

  for (unsigned i = 0; i < resourceInfos.size(); ++i) {
    EXPECT_TRUE(irUseCount[i] > 0)
        << "resourceInfo for deviceId = " << resourceInfos[i].deviceId
        << ", resourceId=" << resourceInfos[i].id << " never used.";
  }
}
