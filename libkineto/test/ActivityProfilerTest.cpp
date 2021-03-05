/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <chrono>

#include "include/libkineto.h"
#include "src/ActivityProfiler.h"
#include "src/ActivityTrace.h"
#include "src/Config.h"
#include "src/CuptiActivityInterface.h"
#include "src/output_base.h"
#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

#define CUDA_LAUNCH_KERNEL CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
#define CUDA_MEMCPY CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = {startTime, endTime, 0, 1, "Test trace", ""};
    gpuOpCount = 0;
  }

  void addOp(std::string name, int64_t startTime, int64_t endTime, int64_t correlation) {
    ClientTraceActivity op;
    op.opType = name;
    op.startTime = startTime;
    op.endTime = endTime;
    op.device = 0;
    op.pthreadId = pthread_self();
    op.sysThreadId = 123;
    op.correlation = correlation;
    activities.push_back(std::move(op));
    span.opCount++;
  }
};

// Provides ability to easily create a few test CUPTI ops
struct MockCuptiActivityBuffer {
  void addRuntimeActivity(
      CUpti_runtime_api_trace_cbid_enum cbid,
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityAPI>(
        start_us, end_us, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
    act.cbid = cbid;
    act.threadId = pthread_self();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addKernelActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityKernel4>(
        start_us, end_us, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    act.deviceId = 0;
    act.streamId = 1;
    act.name = "kernel";
    act.gridX = act.gridY = act.gridZ = 1;
    act.blockX = act.blockY = act.blockZ = 1;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addMemcpyActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityMemcpy>(
        start_us, end_us, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    act.deviceId = 0;
    act.streamId = 2;
    act.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
    act.srcKind = CUPTI_ACTIVITY_MEMORY_KIND_PINNED;
    act.dstKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  template<class T>
  T& createActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    T& act = *static_cast<T*>(malloc(sizeof(T)));
    bzero(&act, sizeof(act));
    act.start = start_us * 1000;
    act.end = end_us * 1000;
    act.correlationId = correlation;
    return act;
  }

  ~MockCuptiActivityBuffer() {
    for (CUpti_Activity* act : activities) {
      free(act);
    }
  }

  std::vector<CUpti_Activity*> activities;
};

// Mock parts of the CuptiActivityInterface
class MockCuptiActivities : public CuptiActivityInterface {
 public:
  virtual int smCount() override {
    return 10;
  }

  virtual const std::pair<int, int> processActivities(
      std::list<CuptiActivityBuffer>& /*unused*/,
      std::function<void(const CUpti_Activity*)> handler) override {
    for (CUpti_Activity* act : activityBuffer->activities) {
      handler(act);
    }
    return {activityBuffer->activities.size(), 100};
  }

  virtual std::unique_ptr<std::list<CuptiActivityBuffer>>
  activityBuffers() override {
    auto list = std::make_unique<std::list<CuptiActivityBuffer>>();
    list->emplace_back(nullptr, 100);
    return list;
  }

  std::unique_ptr<MockCuptiActivityBuffer> activityBuffer;
};


// Common setup / teardown and helper functions
class ActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<ActivityProfiler>(
        cuptiActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
  }

  std::list<CuptiActivityBuffer> createCuptiActivityBuffers() {
    std::list<CuptiActivityBuffer> res;
    res.emplace_back(nullptr, 100);
    return res;
  }

  std::unique_ptr<Config> cfg_;
  MockCuptiActivities cuptiActivities_;
  std::unique_ptr<ActivityProfiler> profiler_;
};


TEST(ActivityProfiler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"ActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  MockCuptiActivities activities;
  ActivityProfiler profiler(activities, /*cpu only*/ true);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;
  bool success = cfg.parse(fmt::format(R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG", filename));

  EXPECT_TRUE(success);
  EXPECT_FALSE(profiler.isActive());

  auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile(), 10);
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
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
}


TEST_F(ActivityProfilerTest, SyncTrace) {
  using ::testing::Return;
  using ::testing::ByMove;

  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"ActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  ActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_us = 100;
  int64_t duration_us = 300;
  auto start_time = time_point<system_clock>(microseconds(start_time_us));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + microseconds(duration_us));

  // Log some cpu ops
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_us, start_time_us + duration_us);
  cpuOps->addOp("op1", 120, 150, 1);
  cpuOps->addOp("op2", 130, 140, 2);
  cpuOps->addOp("op3", 200, 250, 3);
  profiler.transferCpuTrace(std::move(cpuOps));

  // And some GPU ops
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, 133, 138, 1);
  gpuOps->addRuntimeActivity(CUDA_MEMCPY, 210, 220, 2);
  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, 230, 245, 3);
  gpuOps->addKernelActivity(150, 170, 1);
  gpuOps->addMemcpyActivity(240, 250, 2);
  gpuOps->addKernelActivity(260, 320, 3);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler_->reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), cuptiActivities_);
  EXPECT_EQ(trace.activities()->size(), 9);
  std::map<std::string, int> activityCounts;
  std::map<int64_t, int> resourceIds;
  for (auto& activity : *trace.activities()) {
    activityCounts[activity->name()]++;
    resourceIds[activity->resourceId()]++;
  }
  for (const auto& p : activityCounts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["cudaLaunchKernel"], 2);
  EXPECT_EQ(activityCounts["cudaMemcpy"], 1);
  EXPECT_EQ(activityCounts["kernel"], 2);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Pinned -> Device)"], 1);

  // Ops and runtime events are on thread 123
  EXPECT_EQ(resourceIds[123], 6);
  // Kernels are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 2);
  EXPECT_EQ(resourceIds[2], 1);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  trace.save(filename);
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
}
