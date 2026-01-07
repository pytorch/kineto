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

namespace KN = KINETO_NAMESPACE;

TEST(XpuptiProfilerTest, TestEvents) {
  KN::Config cfg;

  std::vector<std::string_view> metrics;

  std::set<KN::ActivityType> activities{
      KN::ActivityType::GPU_MEMCPY,
      KN::ActivityType::GPU_MEMSET,
      KN::ActivityType::CONCURRENT_KERNEL,
      KN::ActivityType::EXTERNAL_CORRELATION};

  const std::vector<std::string> expectedActivities = {
      "Memcpy M2D",
      "Memcpy M2D",
      "Memcpy M2D",
      "Run(sycl::_V1::queue, ...)",
      "Memcpy D2M"};

  const std::vector<std::string> expectedTypes = {
      "gpu_memcpy", "gpu_memcpy", "gpu_memcpy", "kernel", "gpu_memcpy"};

  constexpr unsigned repeatCount = 1;
  auto [pSession, pBuffer] = RunProfilerTest(
      metrics, activities, cfg, repeatCount, expectedActivities, expectedTypes);

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
