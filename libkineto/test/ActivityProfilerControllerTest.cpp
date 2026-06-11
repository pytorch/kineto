/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <cstdlib>

#include <chrono>
#include <fstream>
#include <string>

#include <unistd.h>

#include "include/Config.h"
#include "src/ActivityProfilerController.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

namespace {

std::string logUrlToPath(const std::string& url) {
  const std::string prefix = "file://";
  if (url.substr(0, prefix.size()) == prefix) {
    return url.substr(prefix.size());
  }
  return url;
}

void createTempTraceFile(char* filename) {
  const int fd = mkstemps(filename, 5);
  ASSERT_GE(fd, 0) << "mkstemps failed for " << filename;
  close(fd);
}

bool traceFileHasContent(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  return file.is_open() && file.tellg() > 0;
}

} // namespace

TEST(ActivityProfilerController, PrepareTraceClearsPendingAsyncRequest) {
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  createTempTraceFile(filename);
  Config asyncCfg;
  bool success = asyncCfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          filename));
  ASSERT_TRUE(success);

  // Start an async request via scheduleTrace -- populate asyncRequestConfig_
  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.step();
  controller.acceptConfig(asyncCfg);
  EXPECT_FALSE(controller.isActive());

  // Start a sync trace before the async request starts
  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());
  controller.prepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());

  controller.startTrace();

  // When the sync trace stops, the async request should be completely inactive
  auto trace = controller.stopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // Use .step() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.step();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}

TEST(ActivityProfilerController, IgnoreAsyncRequestsWhileSyncTraceIsActive) {
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  createTempTraceFile(filename);
  Config asyncCfg;
  bool success = asyncCfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 4
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          filename));
  ASSERT_TRUE(success);

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.step();

  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());

  // Start a sync trace
  controller.prepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());
  EXPECT_FALSE(controller.canAcceptConfig());

  // Accept an async config while the sync trace is active
  // This should be blocked at the controller level
  controller.acceptConfig(asyncCfg);
  EXPECT_TRUE(controller.isActive());

  controller.startTrace();
  auto trace = controller.stopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // After sync runs to completion, check that the async request was not
  // accepted.
  // Use .step() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.step();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}

TEST(ActivityProfilerController, PrepareTracePreemptsActiveAsyncRequest) {
  char asyncFilename[] = "/tmp/libkineto_testXXXXXX.json";
  createTempTraceFile(asyncFilename);
  Config asyncCfg;
  bool success = asyncCfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 3
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 4
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          asyncFilename));
  ASSERT_TRUE(success);

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.step();

  // Start an async request, step until it's active
  controller.acceptConfig(asyncCfg);
  EXPECT_FALSE(controller.isActive());
  controller.step();
  EXPECT_FALSE(controller.isActive());
  controller.step();
  EXPECT_TRUE(controller.isActive());
  controller.step();
  EXPECT_TRUE(controller.isActive());

  // Start a sync trace, which should preempt the async request
  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());
  controller.prepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());
  controller.startTrace();
  auto trace = controller.stopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // Use .step() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.step();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}
