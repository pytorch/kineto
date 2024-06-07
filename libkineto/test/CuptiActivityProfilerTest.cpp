/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <strings.h>
#include <time.h>
#include <chrono>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "include/Config.h"
#include "include/output_base.h"
#include "include/time_since_epoch.h"
#include "include/libkineto.h"
#include "src/ActivityTrace.h"
#include "src/CuptiActivityApi.h"
#include "src/CuptiActivityProfiler.h"
#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"
#include "test/MockActivitySubProfiler.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

const std::string kParamCommsCallName = "record_param_comms";
static constexpr auto kCollectiveName = "Collective name";
static constexpr auto kDtype = "dtype";
static constexpr auto kInMsgNelems = "In msg nelems";
static constexpr auto kOutMsgNelems = "Out msg nelems";
static constexpr auto kInSplit = "In split size";
static constexpr auto kOutSplit = "Out split size";
static constexpr auto kGroupSize = "Group size";
static constexpr const char* kProcessGroupName = "Process Group Name";
static constexpr const char* kProcessGroupDesc = "Process Group Description";
static constexpr const char* kGroupRanks = "Process Group Ranks";
static constexpr int32_t kTruncatLength = 30;

#define CUDA_LAUNCH_KERNEL CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
#define CUDA_MEMCPY CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
#define CUDA_STREAM_SYNC CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
#define CUDA_EVENT_SYNC CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020

#define CU_LAUNCH_KERNEL CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
#define CU_LAUNCH_KERNEL_EX CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx

namespace {
const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}
} // namespace

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = TraceSpan(startTime, endTime, "Test trace");
    gpuOpCount = 0;
  }

  void addOp(
      std::string name,
      int64_t startTime,
      int64_t endTime,
      int64_t correlation,
      const std::unordered_map<std::string, std::string>& metadataMap = {}) {
    GenericTraceActivity op(span, ActivityType::CPU_OP, name);
    op.startTime = startTime;
    op.endTime = endTime;
    op.device = systemThreadId();
    op.resource = systemThreadId();
    op.id = correlation;

    for (const auto& [key, val] : metadataMap) {
      op.addMetadata(key, val);
    }

    emplace_activity(std::move(op));
    span.opCount++;
  }
};

// Provides ability to easily create a few test CUPTI ops
struct MockCuptiActivityBuffer {
  void addCorrelationActivity(
      int64_t correlation,
      CUpti_ExternalCorrelationKind externalKind,
      int64_t externalId) {
    auto& act = *(CUpti_ActivityExternalCorrelation*)malloc(
        sizeof(CUpti_ActivityExternalCorrelation));
    act.kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
    act.externalId = externalId;
    act.externalKind = externalKind;
    act.correlationId = correlation;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addRuntimeActivity(
      CUpti_runtime_api_trace_cbid_enum cbid,
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityAPI>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
    act.cbid = cbid;
    act.threadId = threadId();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addDriverActivity(
      CUpti_driver_api_trace_cbid_enum cbid,
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityAPI>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_DRIVER;
    act.cbid = cbid;
    act.threadId = threadId();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addKernelActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityKernel4>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    act.deviceId = 0;
    act.contextId = 0;
    act.streamId = 1;
    act.name = "kernel";
    act.gridX = act.gridY = act.gridZ = 1;
    act.blockX = act.blockY = act.blockZ = 1;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addMemcpyActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityMemcpy>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    act.deviceId = 0;
    act.streamId = 2;
    act.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
    act.srcKind = CUPTI_ACTIVITY_MEMORY_KIND_PINNED;
    act.dstKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addSyncActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation,
      CUpti_ActivitySynchronizationType type, int64_t stream = 1) {
    auto& act = createActivity<CUpti_ActivitySynchronization>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
    act.type = type;
    act.contextId = 0;
    act.streamId = stream;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addCollectiveActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityKernel4>(
        start_ns, end_ns, correlation);
    act.name = "collective_gpu";
    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    act.queued = 0;
    act.deviceId = 0;
    act.contextId = 1;
    act.streamId = 0;
    act.registersPerThread = 32;
    act.staticSharedMemory = 1024;
    act.dynamicSharedMemory = 1024;
    act.gridX = act.gridY = act.gridZ = 1;
    act.blockX = act.blockY = act.blockZ = 1;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  template <class T>
  T& createActivity(int64_t start_ns, int64_t end_ns, int64_t correlation) {
    T& act = *static_cast<T*>(malloc(sizeof(T)));
    bzero(&act, sizeof(act));
    act.start = start_ns;
    act.end = end_ns;
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

// Mock parts of the CuptiActivityApi
class MockCuptiActivities : public CuptiActivityApi {
 public:
  virtual const std::pair<int, size_t> processActivities(
      CuptiActivityBufferMap&, /*unused*/
      std::function<void(const CUpti_Activity*)> handler) override {
    for (CUpti_Activity* act : activityBuffer->activities) {
      handler(act);
    }
    return {activityBuffer->activities.size(), 100};
  }

  virtual std::unique_ptr<CuptiActivityBufferMap> activityBuffers() override {
    auto map = std::make_unique<CuptiActivityBufferMap>();
    auto buf = std::make_unique<CuptiActivityBuffer>(100);
    uint8_t* addr = buf->data();
    (*map)[addr] = std::move(buf);
    return map;
  }

  void bufferRequestedOverride(
      uint8_t** buffer,
      size_t* size,
      size_t* maxNumRecords) {
    this->bufferRequested(buffer, size, maxNumRecords);
  }

  std::unique_ptr<MockCuptiActivityBuffer> activityBuffer;
};

// Common setup / teardown and helper functions
class CuptiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<CuptiActivityProfiler>(
        cuptiActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
    loggerFactory.addProtocol("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<Config> cfg_;
  MockCuptiActivities cuptiActivities_;
  std::unique_ptr<CuptiActivityProfiler> profiler_;
  ActivityLoggerFactory loggerFactory;
};

void checkTracefile(const char* filename) {
#ifdef __linux__
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf {};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}

TEST(CuptiActivityProfiler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  MockCuptiActivities activities;
  CuptiActivityProfiler profiler(activities, /*cpu only*/ true);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;

  int iter = 0;
  int warmup = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  bool success = cfg.parse(fmt::format(
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
  EXPECT_FALSE(profiler.isActive());

  auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());

  // Usually configuration is done when now is startTime - warmup to kick off
  // warmup but start right away in the test
  profiler.configure(cfg, now);
  profiler.setLogger(logger.get());

  EXPECT_TRUE(profiler.isActive());

  // fast forward in time and we have reached the startTime
  now = startTime;

  // Run the profiler
  // Warmup
  // performRunLoopStep is usually called by the controller loop and takes
  // the current time and the controller's next wakeup time.
  profiler.performRunLoopStep(
      /* Current time */ now, /* Next wakeup time */ now);

  auto next = now + milliseconds(1000);

  // performRunLoopStep can also be called by an application thread to update
  // iteration count since this config does not use iteration this should have
  // no effect on the state
  while (++iter < 20) {
    profiler.performRunLoopStep(now, now, iter);
  }

  // Runloop should now be in collect state, so start workload
  // Perform another runloop step, passing in the end profile time as current.
  // This should terminate collection
  profiler.performRunLoopStep(
      /* Current time */ next, /* Next wakeup time */ next);
  // One step needed for each of the Process and Finalize phases
  // Doesn't really matter what times we pass in here.

  EXPECT_TRUE(profiler.isActive());

  auto nextnext = next + milliseconds(1000);

  while (++iter < 40) {
    profiler.performRunLoopStep(next, next, iter);
  }

  EXPECT_TRUE(profiler.isActive());

  profiler.performRunLoopStep(nextnext, nextnext);
  profiler.performRunLoopStep(nextnext, nextnext);

  // Assert that tracing has completed
  EXPECT_FALSE(profiler.isActive());

  checkTracefile(filename);
}

TEST(CuptiActivityProfiler, AsyncTraceUsingIter) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  auto runIterTest = [&](int start_iter, int warmup_iters, int trace_iters) {
    LOG(INFO) << "Async Trace Test: start_iteration = " << start_iter
              << " warmup iterations = " << warmup_iters
              << " trace iterations = " << trace_iters;

    MockCuptiActivities activities;
    CuptiActivityProfiler profiler(activities, /*cpu only*/ true);

    char filename[] = "/tmp/libkineto_testXXXXXX.json";
    mkstemps(filename, 5);

    Config cfg;

    int iter = 0;
    auto now = system_clock::now();

    bool success = cfg.parse(fmt::format(
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
    EXPECT_FALSE(profiler.isActive());

    auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());

    // Usually configuration is done when now is startIter - warmup iter to kick
    // off warmup but start right away in the test
    while (iter < (start_iter - warmup_iters)) {
      profiler.performRunLoopStep(now, now, iter++);
    }

    profiler.configure(cfg, now);
    profiler.setLogger(logger.get());

    EXPECT_TRUE(profiler.isActive());

    // fast forward in time, mimicking what will happen in reality
    now += seconds(10);
    auto next = now + milliseconds(1000);

    // this call to runloop step should not be effecting the state
    profiler.performRunLoopStep(now, next);
    EXPECT_TRUE(profiler.isActive());

    // start trace collection
    while (iter < start_iter) {
      profiler.performRunLoopStep(now, next, iter++);
    }

    // Runloop should now be in collect state, so start workload

    while (iter < (start_iter + trace_iters)) {
      profiler.performRunLoopStep(now, next, iter++);
    }

    // One step is required for each of the Process and Finalize phases
    // Doesn't really matter what times we pass in here.
    if (iter >= (start_iter + trace_iters)) {
      profiler.performRunLoopStep(now, next, iter++);
    }
    EXPECT_TRUE(profiler.isActive());

    auto nextnext = next + milliseconds(1000);

    profiler.performRunLoopStep(nextnext, nextnext);
    profiler.performRunLoopStep(nextnext, nextnext);

    // Assert that tracing has completed
    EXPECT_FALSE(profiler.isActive());

    checkTracefile(filename);
  };

  // start iter = 50, warmup iters = 5, trace iters = 10
  runIterTest(50, 5, 10);
  // should be able to start at 0 iteration
  runIterTest(0, 0, 2);
  runIterTest(0, 5, 5);
}

TEST_F(CuptiActivityProfilerTest, SyncTrace) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

  profiler.recordThreadInfo();

  // Log some cpu ops
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 20, start_time_ns + 50, 1);
  cpuOps->addOp("op2", start_time_ns + 30, start_time_ns + 40, 2);
  cpuOps->addOp("op3", start_time_ns + 100, start_time_ns + 150, 3);
  cpuOps->addOp("op4", start_time_ns + 160, start_time_ns + 180, 4);
  cpuOps->addOp("op5", start_time_ns + 190, start_time_ns + 210, 4);
  profiler.transferCpuTrace(std::move(cpuOps));

  // And some GPU ops
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, start_time_ns + 33, start_time_ns + 38, 1);
  gpuOps->addRuntimeActivity(CUDA_MEMCPY, start_time_ns + 110, start_time_ns + 120, 2);
  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, start_time_ns + 130, start_time_ns + 145, 3);
  gpuOps->addDriverActivity(CU_LAUNCH_KERNEL, start_time_ns + 165, start_time_ns + 175, 4);
  gpuOps->addDriverActivity(CU_LAUNCH_KERNEL_EX, start_time_ns + 195, start_time_ns + 205, 5);
  gpuOps->addRuntimeActivity(CUDA_STREAM_SYNC, start_time_ns + 146, start_time_ns + 240, 6);
  gpuOps->addRuntimeActivity(CUDA_EVENT_SYNC, start_time_ns + 241, start_time_ns + 250, 7);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  gpuOps->addMemcpyActivity(start_time_ns + 140, start_time_ns + 150, 2);
  gpuOps->addKernelActivity(start_time_ns + 160, start_time_ns + 220, 3);
  gpuOps->addKernelActivity(start_time_ns + 230, start_time_ns + 250, 4);
  gpuOps->addKernelActivity(start_time_ns + 260, start_time_ns + 280, 5);
  gpuOps->addSyncActivity(start_time_ns + 221, start_time_ns + 223, 6, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE);
  // Add wait event on kernel stream 1
  gpuOps->addSyncActivity(
      start_time_ns + 224, start_time_ns + 226, 7, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      1 /*stream*/);
  // This event should be ignored because it is not on a stream that has no GPU
  // kernels
  gpuOps->addSyncActivity(
      start_time_ns + 226, start_time_ns + 230, 8, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      4 /*stream*/);
  // Comes from CudaEventSynchronize call on CPU
  gpuOps->addSyncActivity(
      start_time_ns + 227, start_time_ns + 226, 7, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE,
      -1 /*stream*/);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 20);
  std::map<std::string, int> activityCounts;
  std::map<int64_t, int> resourceIds;
  for (auto& activity : *trace.activities()) {
    activityCounts[activity->name()]++;
    resourceIds[activity->resourceId()]++;
    LOG(INFO) << "[test]" << activity->name() << "," << activity->resourceId();
  }
  for (const auto& p : activityCounts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["op4"], 1);
  EXPECT_EQ(activityCounts["cudaLaunchKernel"], 2);
  EXPECT_EQ(activityCounts["cuLaunchKernelEx"], 1);
  EXPECT_EQ(activityCounts["cudaMemcpy"], 1);
  EXPECT_EQ(activityCounts["cudaStreamSynchronize"], 1);
  EXPECT_EQ(activityCounts["cudaEventSynchronize"], 1);
  EXPECT_EQ(activityCounts["kernel"], 4);
  EXPECT_EQ(activityCounts["Stream Sync"], 1);
  EXPECT_EQ(activityCounts["Event Sync"], 1);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Pinned -> Device)"], 1);

  auto sysTid = systemThreadId();
  // Ops and runtime events are on thread sysTid along with the flow start events
  EXPECT_EQ(resourceIds[sysTid], 12);
  // Kernels and sync events are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 6);
  EXPECT_EQ(resourceIds[2], 1);

#ifdef __linux__
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
  struct stat buf {};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
#endif
}

TEST_F(CuptiActivityProfilerTest, GpuNCCLCollectiveTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

  int64_t kernelLaunchTime = start_time_ns + 20;
  profiler.recordThreadInfo();

  // Prepare metadata map
  std::unordered_map<std::string, std::string> metadataMap;
  metadataMap.emplace(kCollectiveName, fmt::format("\"{}\"", "_allgather_base"));
  metadataMap.emplace(kDtype, fmt::format("\"{}\"", "Float"));
  metadataMap.emplace(kInMsgNelems, "65664");
  metadataMap.emplace(kOutMsgNelems, "131328");
  metadataMap.emplace(kGroupSize, "2");
  metadataMap.emplace(kProcessGroupName, fmt::format("\"{}\"", "12341234"));
  metadataMap.emplace(kProcessGroupDesc, fmt::format("\"{}\"", "test_purpose"));

  std::vector<int64_t> inSplitSizes(50, 0);
  std::string inSplitSizesStr = "";
  // Logic is copied from: https://fburl.com/code/811a3wq8
  if (!inSplitSizes.empty() && inSplitSizes.size() <= kTruncatLength) {
    inSplitSizesStr = fmt::format("\"[{}]\"", fmt::join(inSplitSizes, ", "));
    metadataMap.emplace(kInSplit, inSplitSizesStr);
  } else if (inSplitSizes.size() > kTruncatLength) {
    inSplitSizesStr = fmt::format(
        "\"[{}, ...]\"",
        fmt::join(
            inSplitSizes.begin(), inSplitSizes.begin() + kTruncatLength, ", "));
    metadataMap.emplace(kInSplit, inSplitSizesStr);
  }

  std::vector<int64_t> outSplitSizes(20, 1);
  std::string outSplitSizesStr = "";
  // Logic is copied from: https://fburl.com/code/811a3wq8
  if (!outSplitSizes.empty() && outSplitSizes.size() <= kTruncatLength) {
    outSplitSizesStr = fmt::format("\"[{}]\"", fmt::join(outSplitSizes, ", "));
    metadataMap.emplace(kOutSplit, outSplitSizesStr);
  } else if (outSplitSizes.size() > kTruncatLength) {
    outSplitSizesStr = fmt::format(
        "\"[{}, ...]\"",
        fmt::join(
            outSplitSizes.begin(),
            outSplitSizes.begin() + kTruncatLength,
            ", "));
    metadataMap.emplace(kOutSplit, outSplitSizesStr);
  }

  std::vector<int64_t> groupRanks(64, 0);
  std::string groupRanksStr = "";
  if (!groupRanks.empty() && groupRanks.size() <= kTruncatLength) {
    metadataMap.emplace(
        kGroupRanks, fmt::format("\"[{}]\"", fmt::join(groupRanks, ", ")));
  } else if (groupRanks.size() > kTruncatLength) {
    metadataMap.emplace(
        kGroupRanks,
        fmt::format(
            "\"[{}, ..., {}]\"",
            fmt::join(
                groupRanks.begin(),
                groupRanks.begin() + kTruncatLength - 1,
                ", "),
            groupRanks.back()));
  }

  // Set up CPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp(
      kParamCommsCallName,
      kernelLaunchTime,
      kernelLaunchTime + 10,
      1,
      metadataMap);
  profiler.transferCpuTrace(std::move(cpuOps));

  // Set up corresponding GPU events and connect with CPU events
  // via correlationId
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 1);
  gpuOps->addCollectiveActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the content of GPU event and we should see extra
  // collective fields get populated from CPU event.
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(2, trace.activities()->size());
  auto& cpu_annotation = trace.activities()->at(0);
  auto& gpu_annotation = trace.activities()->at(1);
  EXPECT_EQ(cpu_annotation->name(), kParamCommsCallName);
  EXPECT_EQ(gpu_annotation->name(), "collective_gpu");

  // Check vector with length > 30 get truncated successfully
  std::vector<int64_t> expectedInSplit(kTruncatLength, 0);
  auto expectedInSplitStr =
      fmt::format("\"[{}, ...]\"", fmt::join(expectedInSplit, ", "));
  EXPECT_EQ(cpu_annotation->getMetadataValue(kInSplit), expectedInSplitStr);
  std::vector<int64_t> expectedGroupRanks(kTruncatLength - 1, 0);
  auto expectedGroupRanksStr = fmt::format(
      "\"[{}, ..., {}]\"", fmt::join(expectedGroupRanks, ", "), "0");
  EXPECT_EQ(
      cpu_annotation->getMetadataValue(kGroupRanks), expectedGroupRanksStr);

#ifdef __linux__
  // Test saved output can be loaded as JSON
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  LOG(INFO) << "Logging to tmp file: " << filename;
  trace.save(filename);

  // Check that the saved JSON file can be loaded and deserialized
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  folly::dynamic jsonData = folly::parseJson(jsonStr);

  // Convert the folly::dynamic object to a string and check
  // if the substring exists
  std::string jsonString = folly::toJson(jsonData);
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

  EXPECT_EQ(2, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(2, countSubstrings(jsonString, kInMsgNelems));
  EXPECT_EQ(2, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(2, countSubstrings(jsonString, kOutMsgNelems));
  EXPECT_EQ(2, countSubstrings(jsonString, "131328"));
  EXPECT_EQ(2, countSubstrings(jsonString, kInSplit));
  EXPECT_EQ(2, countSubstrings(jsonString, expectedInSplitStr));
  EXPECT_EQ(2, countSubstrings(jsonString, kOutSplit));
  EXPECT_EQ(2, countSubstrings(jsonString, outSplitSizesStr));
  EXPECT_EQ(2, countSubstrings(jsonString, kCollectiveName));
  EXPECT_EQ(2, countSubstrings(jsonString, "_allgather_base"));
  EXPECT_EQ(2, countSubstrings(jsonString, kProcessGroupName));
  EXPECT_EQ(2, countSubstrings(jsonString, "12341234"));
  EXPECT_EQ(2, countSubstrings(jsonString, kProcessGroupDesc));
  EXPECT_EQ(2, countSubstrings(jsonString, "test_purpose"));
  EXPECT_EQ(2, countSubstrings(jsonString, kGroupRanks));
  EXPECT_EQ(2, countSubstrings(jsonString, expectedGroupRanksStr));
#endif
}

TEST_F(CuptiActivityProfilerTest, GpuUserAnnotationTest) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

  int64_t kernelLaunchTime = start_time_ns + 20;
  profiler.recordThreadInfo();

  // set up CPU event
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("annotation", kernelLaunchTime, kernelLaunchTime + 10, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // set up a couple of GPU events and correlate with above CPU event.
  // CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1 is used for user annotations.
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 15, kernelLaunchTime + 25, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  std::map<std::string, int> counts;
  for (auto& activity : *trace.activities()) {
    counts[activity->name()]++;
  }

  // We should now have an additional annotation activity created
  // on the GPU timeline.
  EXPECT_EQ(counts["annotation"], 2);
  EXPECT_EQ(counts["kernel"], 2);

  auto& annotation = trace.activities()->at(0);
  auto& kernel1 = trace.activities()->at(1);
  auto& kernel2 = trace.activities()->at(2);
  auto& gpu_annotation = trace.activities()->at(3);
  EXPECT_EQ(gpu_annotation->type(), ActivityType::GPU_USER_ANNOTATION);
  EXPECT_EQ(gpu_annotation->timestamp(), kernel1->timestamp());
  EXPECT_EQ(
      gpu_annotation->duration(),
      kernel2->timestamp() + kernel2->duration() - kernel1->timestamp());
  EXPECT_EQ(gpu_annotation->deviceId(), kernel1->deviceId());
  EXPECT_EQ(gpu_annotation->resourceId(), kernel1->resourceId());
  EXPECT_EQ(gpu_annotation->correlationId(), annotation->correlationId());
  EXPECT_EQ(gpu_annotation->name(), annotation->name());
}

TEST_F(CuptiActivityProfilerTest, SubActivityProfilers) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Setup example events to test
  GenericTraceActivity ev{defaultTraceSpan(), ActivityType::GLOW_RUNTIME, ""};
  ev.device = 1;
  ev.resource = 0;

  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 1000;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));

  std::deque<GenericTraceActivity> test_activities{3, ev};
  test_activities[0].startTime = start_time_ns;
  test_activities[0].endTime = start_time_ns + 5000;
  test_activities[0].activityName = "SubGraph A execution";
  test_activities[1].startTime = start_time_ns;
  test_activities[1].endTime = start_time_ns + 2000;
  test_activities[1].activityName = "Operator foo";
  test_activities[2].startTime = start_time_ns + 2500;
  test_activities[2].endTime = start_time_ns + 2900;
  test_activities[2].activityName = "Operator bar";

  auto mock_activity_profiler =
      std::make_unique<MockActivityProfiler>(test_activities);

  MockCuptiActivities activities;
  CuptiActivityProfiler profiler(activities, /*cpu only*/ true);
  profiler.addChildActivityProfiler(std::move(mock_activity_profiler));

  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  EXPECT_TRUE(profiler.isActive());

  profiler.stopTrace(start_time + nanoseconds(duration_ns));
  EXPECT_TRUE(profiler.isActive());

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  LOG(INFO) << "Logging to tmp file " << filename;

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  trace.save(filename);
  const auto& traced_activites = trace.activities();

  // Test we have all the events
  EXPECT_EQ(traced_activites->size(), test_activities.size());

  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);

  // Should expect at least 100 bytes
  struct stat buf {};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
}

TEST_F(CuptiActivityProfilerTest, BufferSizeLimitTestWarmup) {
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);

  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  int maxBufferSizeMB = 3;

  auto startTimeEpoch = std::to_string(
      duration_cast<milliseconds>(startTime.time_since_epoch()).count());
  std::string maxBufferSizeMBStr = std::to_string(maxBufferSizeMB);
  cfg_->handleOption("ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB", maxBufferSizeMBStr);
  cfg_->handleOption("PROFILE_START_TIME", startTimeEpoch);

  EXPECT_FALSE(profiler.isActive());
  profiler.configure(*cfg_, now);
  EXPECT_TRUE(profiler.isActive());

  for (size_t i = 0; i < maxBufferSizeMB; i++) {
    uint8_t* buf;
    size_t gpuBufferSize;
    size_t maxNumRecords;
    cuptiActivities_.bufferRequestedOverride(
        &buf, &gpuBufferSize, &maxNumRecords);
  }

  // fast forward to startTime and profiler is now running
  now = startTime;

  profiler.performRunLoopStep(now, now);

  auto next = now + milliseconds(1000);
  profiler.performRunLoopStep(next, next);
  profiler.performRunLoopStep(next, next);
  profiler.performRunLoopStep(next, next);

  EXPECT_FALSE(profiler.isActive());
}

TEST(CuptiActivityProfiler, MetadataJsonFormatingTest) {
  // Check for Json string sanitation
  // based on AsyncTrace test
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  MockCuptiActivities activities;
  CuptiActivityProfiler profiler(activities, /*cpu only*/ true);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  Config cfg;

  auto now = system_clock::now();
  auto startTime = now + seconds(2);

  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 1
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      filename,
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(profiler.isActive());

  auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());

  // Usually configuration is done when now is startTime - warmup to kick off
  // warmup but start right away in the test
  profiler.configure(cfg, now);
  profiler.setLogger(logger.get());

  EXPECT_TRUE(profiler.isActive());

  // Add test metadata
  std::string keyPrefix = "TEST_METADATA_";
  profiler.addMetadata(keyPrefix + "NORMAL", "\"metadata value\"");
  profiler.addMetadata(keyPrefix + "NEWLINE", "\"metadata \nvalue\"");
  profiler.addMetadata(keyPrefix + "BACKSLASH", "\"/test/metadata\\path\"");

  // Profiling activity at start up/during active/after duration
  auto next = startTime + milliseconds(1000);
  auto after = next + milliseconds(1000);

  profiler.performRunLoopStep(startTime, startTime);
  EXPECT_TRUE(profiler.isActive());

  profiler.performRunLoopStep(next, next);
  EXPECT_TRUE(profiler.isActive());

  profiler.performRunLoopStep(after, after);
  EXPECT_FALSE(profiler.isActive());

#ifdef __linux__
  // Check that the saved JSON file can be loaded and deserialized
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

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

  // Check if metadata has been correctly sanitized
  EXPECT_EQ(3, countSubstrings(jsonStr, keyPrefix));
  EXPECT_EQ(2, countSubstrings(jsonStr, "metadata value"));
  EXPECT_EQ(1, countSubstrings(jsonStr, "/test/metadata/path"));
#endif
}

TEST_F(CuptiActivityProfilerTest, JsonGPUIDSortTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));
  profiler.recordThreadInfo();

  // Set up CPU events and corresponding GPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
  profiler.transferCpuTrace(std::move(cpuOps));
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, start_time_ns + 23, start_time_ns + 28, 1);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the content of GPU event and we should see extra
  // collective fields get populated from CPU event.
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(3, trace.activities()->size());

#ifdef __linux__
  // Test saved output can be loaded as JSON
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  LOG(INFO) << "Logging to tmp file: " << filename;
  trace.save(filename);

  // Check that the saved JSON file can be loaded and deserialized
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  folly::dynamic jsonData = folly::parseJson(jsonStr);

  std::unordered_map<int64_t, std::string> sortLabel;
  std::unordered_map<int64_t, int64_t> sortIdx;
  for (auto& event : jsonData["traceEvents"]) {
    if (event["name"] == "process_labels" && event["tid"] == 0 && event["pid"].isInt()) {
      sortLabel[event["pid"].asInt()] = event["args"]["labels"].asString();
    }
    if (event["name"] == "process_sort_index" && event["tid"] == 0 && event["pid"].isInt()) {
      sortIdx[event["pid"].asInt()] = event["args"]["sort_index"].asInt();
    }
  }

  // Expect there is 1 CUPTI Overhead, and 16 CPU + GPU sorts, total 17.
  EXPECT_EQ(17, sortLabel.size());
  for (int i = 0; i<16; i++) {
    // Check there are 16 GPU sorts (0-15) with expected sort_index.
    EXPECT_EQ("GPU " + std::to_string(i), sortLabel[i]);
    // sortIndex is gpu + kExceedMaxPid to put GPU tracks at the bottom
    // of the trace timelines.
    EXPECT_EQ(i + kExceedMaxPid, sortIdx[i]);
  }
#endif
}
