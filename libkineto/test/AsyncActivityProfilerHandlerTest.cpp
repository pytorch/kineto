/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <folly/json/json.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <chrono>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "include/Config.h"
#include "src/AsyncActivityProfilerHandler.h"
#include "src/GenericActivityProfiler.h"
#include "src/SyncActivityProfilerHandler.h"

#include "src/Logger.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

// Subclass that lets us control isGpuCollectionStopped() for testing
// the buffer-overflow-during-warmup path without needing real CUPTI.
class MockGpuProfiler : public GenericActivityProfiler {
 public:
  MockGpuProfiler() : GenericActivityProfiler(/*cpuOnly=*/false) {}

  void setGpuCollectionStopped(bool stopped) {
    gpuStopped_ = stopped;
  }

 protected:
  bool isGpuCollectionStoppedImpl() const override {
    return gpuStopped_;
  }

 private:
  bool gpuStopped_{false};
};

static std::string logUrlToPath(const std::string& url) {
  const std::string prefix = "file://";
  if (url.substr(0, prefix.size()) == prefix) {
    return url.substr(prefix.size());
  }
  return url;
}

static void checkTracefile(const char* filename) {
#ifdef __linux__
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}

TEST(AsyncActivityProfilerHandler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  GenericActivityProfiler profiler(/*cpu only*/ true);
  std::atomic_bool syncTraceActive{false};
  AsyncActivityProfilerHandler handler(profiler, syncTraceActive);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;

  int iter = 0;
  int warmup = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          warmup,
          filename,
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  // fast forward in time and we have reached the startTime
  now = startTime;

  // Warmup
  handler.performRunLoopStep(now, now);

  auto next = now + milliseconds(1000);

  // Iteration-based steps should have no effect on timestamp-based config
  while (++iter < 20) {
    handler.performRunLoopStep(now, now, iter);
  }

  // Terminate collection
  handler.performRunLoopStep(next, next);

  EXPECT_TRUE(handler.isAsyncActive());

  auto nextnext = next + milliseconds(1000);

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
    std::atomic_bool syncTraceActive{false};
    AsyncActivityProfilerHandler handler(profiler, syncTraceActive);

    char filename[] = "/tmp/libkineto_testXXXXXX.json";
    mkstemps(filename, 5);

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
            filename));

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
  std::atomic_bool syncTraceActive{false};
  AsyncActivityProfilerHandler handler(profiler, syncTraceActive);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;

  auto now = system_clock::now();
  auto startTime = now + seconds(2);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 1
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          filename,
          duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  std::string keyPrefix = "TEST_METADATA_";
  profiler.addMetadata(keyPrefix + "NORMAL", "\"metadata value\"");
  profiler.addMetadata(keyPrefix + "NEWLINE", "\"metadata \nvalue\"");
  profiler.addMetadata(keyPrefix + "BACKSLASH", R"("/test/metadata\path")");

  auto next = startTime + milliseconds(1000);
  auto after = next + milliseconds(1000);

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
  folly::dynamic jsonData = folly::parseJson(jsonStr);

  auto countSubstrings = [](const std::string& source,
                            const std::string& substring) {
    size_t count = 0;
    size_t pos = source.find(substring);
    while (pos != std::string::npos) {
      ++count;
      pos = source.find(substring, pos + substring.length());
    }
    return count;
  };

  EXPECT_EQ(3, countSubstrings(jsonStr, keyPrefix));
  EXPECT_EQ(2, countSubstrings(jsonStr, "metadata value"));
  EXPECT_EQ(1, countSubstrings(jsonStr, "/test/metadata/path"));

  EXPECT_EQ(jsonData["PT_PROFILER_JOB_NAME"].asString(), "test_training_job");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_VERSION"].asString(), "2");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_ATTEMPT_INDEX"].asString(), "5");
#endif

  unsetenv("PT_PROFILER_JOB_NAME");
  unsetenv("PT_PROFILER_JOB_VERSION");
  unsetenv("PT_PROFILER_JOB_ATTEMPT_INDEX");
}

TEST(AsyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  std::atomic_bool syncTraceActive{false};
  AsyncActivityProfilerHandler handler(profiler, syncTraceActive);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;
  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 5
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
          filename,
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

TEST(AsyncActivityProfilerHandler, BufferSizeLimitDuringWarmup) {
  MockGpuProfiler profiler;
  std::atomic_bool syncTraceActive{false};
  AsyncActivityProfilerHandler handler(profiler, syncTraceActive);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;
  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  bool success = cfg.parse(
      fmt::format(
          R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 5
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
    ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB = 3
  )CFG",
          filename,
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

TEST(SyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  std::atomic_bool syncTraceActive{false};
  SyncActivityProfilerHandler handler(profiler, syncTraceActive);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());

  // Cancel after prepareTrace
  Config cfg;
  cfg.validate(system_clock::now());
  handler.prepareTrace(cfg);
  EXPECT_TRUE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());
}
