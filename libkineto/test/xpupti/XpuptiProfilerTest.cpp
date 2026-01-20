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
