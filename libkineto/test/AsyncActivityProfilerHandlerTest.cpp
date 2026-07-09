/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cstdlib>

#include <chrono>
#include <fstream>

#include "include/Config.h"
#include "src/AsyncActivityProfilerHandler.h"
#include "src/GenericActivityProfiler.h"

#include "src/Logger.h"
#include "test/TestUtils.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;
using namespace libkineto::test;

// Several tests step to timestamps derived from the warmup and duration values
// they pass in the config, using named constants shared between the two so they
// cannot drift apart. startTime leads now by the warmup period plus a
// one-second buffer: canStart() requires at least ACTIVITIES_WARMUP_PERIOD_SECS
// of lead time, and the buffer absorbs the millisecond truncation in the
// PROFILE_START_TIME round-trip (a bare now + warmup can round just under and
// be rejected). Steps meant to fall after collection add a second past
// startTime + ACTIVITIES_DURATION_SECS.

// Subclass that lets us control isGpuCollectionStopped() for testing
// the buffer-overflow-during-warmup path without needing real CUPTI.
class MockGpuProfiler : public GenericActivityProfiler {
 public:
  MockGpuProfiler() : GenericActivityProfiler(/*cpuOnly=*/false) {}

  void setGpuCollectionStopped(bool stopped) {
    gpuStopped_ = stopped;
  }

 protected:
  bool isGpuCollectionStopped() const override {
    return gpuStopped_;
  }

 private:
  bool gpuStopped_{false};
};

TEST(AsyncActivityProfilerHandler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  constexpr int kWarmupSecs = 5;
  constexpr int kDurationSecs = 1;
  int iter = 0;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          kWarmupSecs,
          kDurationSecs,
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  // fast forward in time and we have reached the startTime
  now = startTime;

  // Warmup
  handler.performRunLoopStep(now, now);

  // End of the collection window: startTime + duration.
  auto next = startTime + seconds(kDurationSecs);

  // Iteration-based steps should have no effect on timestamp-based config
  while (++iter < 20) {
    handler.performRunLoopStep(now, now, iter);
  }

  // Terminate collection
  handler.performRunLoopStep(next, next);

  EXPECT_TRUE(handler.isAsyncActive());

  // Past the window, to drive the finalize (ProcessTrace) steps.
  auto nextnext = next + seconds(kDurationSecs);

  while (++iter < 40) {
    handler.performRunLoopStep(next, next, iter);
  }

  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(nextnext, nextnext);
  handler.performRunLoopStep(nextnext, nextnext);

  EXPECT_FALSE(handler.isAsyncActive());

  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  checkTracefile(logFile.c_str());
}

TEST(AsyncActivityProfilerHandler, AsyncTraceUsingIter) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  auto runIterTest = [&](int start_iter, int warmup_iters, int trace_iters) {
    LOG(INFO) << "Async Trace Test: start_iteration = " << start_iter
              << " warmup iterations = " << warmup_iters
              << " trace iterations = " << trace_iters;

    GenericActivityProfiler profiler(/*cpu only*/ true);
    AsyncActivityProfilerHandler handler(profiler);

    auto traceFile = createTempTraceFile("libkineto_test", ".json");

    Config cfg;

    int iter = 0;
    auto now = system_clock::now();

    bool success = cfg.parse(
        fmt::format(
            R"CFG(
      PROFILE_START_ITERATION = {}
      ACTIVITIES_WARMUP_ITERATIONS={}
      ACTIVITIES_ITERATIONS={}
      ACTIVITIES_DURATION_SECS = 1
      ACTIVITIES_LOG_FILE = {}
    )CFG",
            start_iter,
            warmup_iters,
            trace_iters,
            traceFile.path()));

    EXPECT_TRUE(success);
    EXPECT_FALSE(handler.isAsyncActive());

    while (iter < (start_iter - warmup_iters)) {
      iter++;
    }

    handler.configure(cfg, now);
    EXPECT_TRUE(handler.isAsyncActive());

    now += seconds(10);
    auto next = now + milliseconds(1000);

    handler.performRunLoopStep(now, next);
    EXPECT_TRUE(handler.isAsyncActive());

    while (iter < start_iter) {
      handler.performRunLoopStep(now, next, iter++);
    }

    while (iter < (start_iter + trace_iters)) {
      handler.performRunLoopStep(now, next, iter++);
    }

    if (iter >= (start_iter + trace_iters)) {
      handler.performRunLoopStep(now, next, iter++);
    }
    EXPECT_TRUE(handler.isAsyncActive());

    auto nextnext = next + milliseconds(1000);
    handler.performRunLoopStep(nextnext, nextnext);
    handler.ensureCollectTraceDone();
    handler.performRunLoopStep(nextnext, nextnext);

    EXPECT_FALSE(handler.isAsyncActive());

    auto logFile = logUrlToPath(cfg.activitiesLogUrl());
    checkTracefile(logFile.c_str());
  };

  runIterTest(50, 5, 10);
  runIterTest(0, 0, 2);
  runIterTest(0, 5, 5);
}

TEST(AsyncActivityProfilerHandler, MetadataJsonFormatingTest) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  setenv("PT_PROFILER_JOB_NAME", "test_training_job", 1);
  setenv("PT_PROFILER_JOB_VERSION", "2", 1);
  setenv("PT_PROFILER_JOB_ATTEMPT_INDEX", "5", 1);

  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  constexpr int kWarmupSecs = 1;
  constexpr int kDurationSecs = 1;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          kWarmupSecs,
          kDurationSecs,
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  std::string keyPrefix = "TEST_METADATA_";
  profiler.addMetadata(keyPrefix + "NORMAL", "\"metadata value\"");
  profiler.addMetadata(keyPrefix + "NEWLINE", "\"metadata \nvalue\"");
  profiler.addMetadata(keyPrefix + "BACKSLASH", R"("/test/metadata\path")");

  // End of the collection window, then a step past it to finalize.
  auto next = startTime + seconds(kDurationSecs);
  auto after = next + seconds(kDurationSecs);

  handler.performRunLoopStep(startTime, startTime);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(next, next);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(after, after);
  EXPECT_FALSE(handler.isAsyncActive());

#ifdef __linux__
  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  std::ifstream file(logFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

  EXPECT_EQ(3, countSubstrings(jsonStr, keyPrefix));
  EXPECT_EQ(2, countSubstrings(jsonStr, "metadata value"));
  EXPECT_EQ(1, countSubstrings(jsonStr, "/test/metadata/path"));

  EXPECT_EQ(
      jsonData["PT_PROFILER_JOB_NAME"].get<std::string>(), "test_training_job");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_VERSION"].get<std::string>(), "2");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_ATTEMPT_INDEX"].get<std::string>(), "5");
#endif

  unsetenv("PT_PROFILER_JOB_NAME");
  unsetenv("PT_PROFILER_JOB_VERSION");
  unsetenv("PT_PROFILER_JOB_ATTEMPT_INDEX");
}

TEST(AsyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  constexpr int kWarmupSecs = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          kWarmupSecs,
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  EXPECT_TRUE(success);

  // Cancel during Warmup
  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());

  // Stays inactive on subsequent steps
  handler.performRunLoopStep(startTime, startTime);
  EXPECT_FALSE(handler.isAsyncActive());

  // Cancel during CollectTrace
  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());
  now = startTime;
  handler.performRunLoopStep(now, now);
  EXPECT_TRUE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());
}

TEST(AsyncActivityProfilerHandler, FinalizesPendingTraceOnTeardown) {
  // Destroying a scheduled handler with a collected-but-unprocessed trace must
  // let the profiler loop finalize it, not drop it.
  GenericActivityProfiler profiler(/*cpu only*/ true);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 1
    ACTIVITIES_WARMUP_ITERATIONS = 0
    ACTIVITIES_ITERATIONS = 1
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          traceFile.path()));
  EXPECT_TRUE(success);

  {
    AsyncActivityProfilerHandler handler(profiler);
    handler.step();
    EXPECT_FALSE(handler.isAsyncActive());

    handler.scheduleTrace(cfg);

    // Activate the scheduled config and enter CollectTrace.
    handler.step();
    EXPECT_TRUE(handler.isAsyncActive());

    // CollectTrace -> ProcessTrace asynchronously. Application step()
    // deliberately does not finalize ProcessTrace; the loop thread should do
    // that while exiting.
    handler.step();
    handler.ensureCollectTraceDone();
    EXPECT_TRUE(handler.isAsyncActive());

    // handler goes out of scope here with a collected trace still pending.
  }

  // The pending trace must have been finalized during teardown.
  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  checkTracefile(logFile.c_str());
}

TEST(AsyncActivityProfilerHandler, BufferSizeLimitDuringWarmup) {
  MockGpuProfiler profiler;
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  constexpr int kWarmupSecs = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
    ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB = 3
  )CFG",
          kWarmupSecs,
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  EXPECT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());

  // Simulate GPU buffer overflow
  profiler.setGpuCollectionStopped(true);

  // During warmup, the handler should detect GPU collection stopped
  // and transition back to WaitForRequest
  now = startTime;
  handler.performRunLoopStep(now, now);
  EXPECT_FALSE(handler.isAsyncActive());
}

// An iteration-based request that arrives before the application has begun
// counting iterations (iterationCount_ == -1) cannot be honored unless it also
// carries a duration. With no duration it must be dropped, not scheduled.
TEST(AsyncActivityProfilerHandler, IterationRequestWithoutStepIsRejected) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 0
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          traceFile.path()));
  ASSERT_TRUE(success);
  ASSERT_TRUE(cfg.hasProfileStartIteration());

  // No step() has advanced the iteration count, so the request is dropped
  // synchronously.
  EXPECT_FALSE(handler.scheduleTrace(cfg));
  EXPECT_FALSE(handler.isAsyncActive());
}

// The same iteration-based request is accepted when it carries a duration: the
// handler falls back to duration/timestamp scheduling instead of rejecting it.
// This is the path daemon configs take when the application is not reporting
// iterations.
TEST(
    AsyncActivityProfilerHandler,
    IterationRequestWithDurationFallsBackToTimestamp) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
          traceFile.path()));
  ASSERT_TRUE(success);
  ASSERT_TRUE(cfg.hasProfileStartIteration());

  // Accepted via the duration fallback (contrast the no-duration case above,
  // which is rejected). scheduleTrace() reports the decision synchronously.
  EXPECT_TRUE(handler.scheduleTrace(cfg));
}

// Only one on-demand request may be pending at a time. A second request that
// arrives while the first is still pending is dropped.
TEST(AsyncActivityProfilerHandler, SecondRequestWhilePendingIsRejected) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto acceptedFile = createTempTraceFile("libkineto_test_accepted", ".json");
  auto rejectedFile = createTempTraceFile("libkineto_test_rejected", ".json");

  // Start far in the future so neither request activates during the test; the
  // accept/reject decision is made synchronously inside scheduleTrace().
  auto startMs = duration_cast<milliseconds>(
                     (system_clock::now() + seconds(3600)).time_since_epoch())
                     .count();

  Config accepted;
  ASSERT_TRUE(accepted.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          acceptedFile.path(),
          startMs)));

  Config rejected;
  ASSERT_TRUE(rejected.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          rejectedFile.path(),
          startMs)));

  // The first request is accepted; a second arriving while it is still pending
  // is rejected.
  EXPECT_TRUE(handler.scheduleTrace(accepted));
  EXPECT_FALSE(handler.scheduleTrace(rejected));
}

// configure() must refuse a request whose start time has already passed rather
// than entering warmup for a trace that can never start on time. The start time
// is kept within the max request age so the config still parses.
TEST(AsyncActivityProfilerHandler, ConfigureRejectsStartTimeInThePast) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  auto now = system_clock::now();
  auto startTime = now - seconds(5);

  Config cfg;
  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 5
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  ASSERT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_FALSE(handler.isAsyncActive());
}

// cancel() must tear down cleanly when the trace has finished collecting and is
// waiting to be processed (RunloopState::ProcessTrace), returning the handler
// to the idle state.
TEST(AsyncActivityProfilerHandler, CancelDuringProcessTrace) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  constexpr int kWarmupSecs = 5;
  constexpr int kDurationSecs = 1;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  Config cfg;
  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          kWarmupSecs,
          kDurationSecs,
          traceFile.path(),
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  ASSERT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());

  // Warmup -> CollectTrace at the start time.
  now = startTime;
  handler.performRunLoopStep(now, now);

  // CollectTrace -> ProcessTrace once the collection window closes (a second
  // past startTime + duration). Driving with the default currentIter (-1)
  // collects inline, so the state settles on ProcessTrace.
  auto afterEnd = startTime + seconds(kDurationSecs + 1);
  handler.performRunLoopStep(afterEnd, afterEnd);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());
}

// acceptConfig() only schedules when the config actually enables the activity
// profiler; a config with activities disabled is a no-op.
TEST(
    AsyncActivityProfilerHandler,
    AcceptConfigIgnoresDisabledActivityProfiler) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  Config cfg;
  ASSERT_TRUE(cfg.parse("ACTIVITIES_ENABLED = false"));
  ASSERT_FALSE(cfg.activityProfilerEnabled());

  // acceptConfig() reports that it declined the disabled config, rather than
  // silently doing nothing.
  EXPECT_FALSE(handler.acceptConfig(cfg));
  EXPECT_FALSE(handler.isAsyncActive());
}

// acceptConfig() schedules and reports acceptance when the config enables the
// activity profiler. Paired with the disabled case above, this pins that the
// return value reflects the config rather than being constant.
TEST(
    AsyncActivityProfilerHandler,
    AcceptConfigSchedulesEnabledActivityProfiler) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  // Far-future start so the request stays scheduled without activating during
  // the test.
  auto startMs = duration_cast<milliseconds>(
                     (system_clock::now() + seconds(3600)).time_since_epoch())
                     .count();

  Config cfg;
  ASSERT_TRUE(cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          traceFile.path(),
          startMs)));
  ASSERT_TRUE(cfg.activityProfilerEnabled());

  EXPECT_TRUE(handler.acceptConfig(cfg));
}
