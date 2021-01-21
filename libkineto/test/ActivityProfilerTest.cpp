/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include "include/libkineto.h"
#include "src/ActivityProfiler.h"
#include "src/Config.h"
#include "src/CuptiActivityInterface.h"
#include "src/output_json.h"

#include "src/Logger.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

class MockCuptiActivities : public CuptiActivityInterface {};

TEST(ActivityProfiler, PyTorchTrace) {
  std::vector<std::string> log_modules(
      {"ActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  MockCuptiActivities activities;
  ActivityProfiler profiler(activities, /*cpu only*/ true);

  Config cfg;
  bool success = cfg.parse(R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
  )CFG");

  EXPECT_TRUE(success);
  EXPECT_FALSE(profiler.isActive());

  auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());
  auto now = system_clock::now();
  profiler.configure(cfg, now);
  profiler.setLogger(logger.get());

  EXPECT_TRUE(profiler.isActive());

  // Run the profiler
  // Warmup
  // performRunLoopStep is usually called by the controller loop and takes
  // the current time and the controller's next wakeup time.
  profiler.performRunLoopStep(
      /* Current time */ now, /* Next wakeup time */ now);

  // Runloop should now be in collect state, so start workload
  auto next = now + milliseconds(1000);
  // Perform another runloop step, passing in the end profile time as current.
  // This should terminate collection
  profiler.performRunLoopStep(
      /* Current time */ next, /* Next wakeup time */ next);
  // One step needed for each of the Process and Finalize phases
  // Doesn't really matter what times we pass in here.
  profiler.performRunLoopStep(next, next);
  profiler.performRunLoopStep(next, next);

  // Assert that tracing has completed
  EXPECT_FALSE(profiler.isActive());

  // Check that the expected file was written and that it has some content
  pid_t curr_pid = getpid();
  const std::string filename =
      fmt::format("/tmp/libkineto_activities_{}.json", curr_pid);
  struct stat buf;
  bool logfile_exists = stat(filename.c_str(), &buf) == 0;
  if (!logfile_exists) {
    perror(filename.c_str());
  }
  EXPECT_TRUE(logfile_exists);
  // Should expect at least 1MB
  EXPECT_GT(buf.st_size, 100);
}
