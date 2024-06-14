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
#include <cstdlib>

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
#include "src/CuptiActivityProfiler.h"
#include "src/RoctracerActivityApi.h"
#include "src/RoctracerLogger.h"
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

#define HIP_LAUNCH_KERNEL HIP_API_ID_hipLaunchKernel
#define HIP_MEMCPY HIP_API_ID_hipMemcpy
#define HIP_MALLOC HIP_API_ID_hipMalloc
#define HIP_FREE HIP_API_ID_hipFree

namespace {
const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}
} // namespace

// Provides ability to easily create a test CPU-side ops
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

// Provides ability to easily create a test Roctracer ops
struct MockRoctracerLogger {
  void addCorrelationActivity(
      uint64_t correlation,
      RoctracerLogger::CorrelationDomain domain,
      uint64_t externalId) {
    externalCorrelations_[domain].emplace_back(correlation, externalId);
  }

  void addRuntimeKernelActivity(
      uint32_t cid, int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerKernelRow* row = new roctracerKernelRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      cid,
      processId(),
      systemThreadId(),
      start_ns,
      end_ns,
      nullptr,
      nullptr,
      0,0,0,0,0,0,0,0
    );
    activities_.push_back(row);
  }

  void addRuntimeMallocActivity(
      uint32_t cid, int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerMallocRow* row = new roctracerMallocRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      cid,
      processId(),
      systemThreadId(),
      start_ns,
      end_ns,
      nullptr,
      1
    );
    activities_.push_back(row);
  }

  void addRuntimeCopyActivity(
      uint32_t cid, int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerCopyRow* row = new roctracerCopyRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      cid,
      processId(),
      systemThreadId(),
      start_ns,
      end_ns,
      nullptr,
      nullptr,
      1,
      hipMemcpyHostToHost,
      static_cast<hipStream_t>(0)
    );
    activities_.push_back(row);
  }

  void addKernelActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerAsyncRow* row = new roctracerAsyncRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      HIP_OP_DISPATCH_KIND_KERNEL_,
      0,
      0,
      1,
      start_ns,
      end_ns,
      std::string("kernel")
    );
    activities_.push_back(row);
  }

  void addMemcpyH2DActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerAsyncRow* row = new roctracerAsyncRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      HIP_OP_COPY_KIND_HOST_TO_DEVICE_,
      0,
      0,
      2,
      start_ns,
      end_ns,
      std::string()
    );
    activities_.push_back(row);
  }

  void addMemcpyD2HActivity(
      int64_t start_ns, int64_t end_ns, int64_t correlation) {
    roctracerAsyncRow* row = new roctracerAsyncRow(
      correlation,
      ACTIVITY_DOMAIN_HIP_API,
      HIP_OP_COPY_KIND_DEVICE_TO_HOST_,
      0,
      0,
      2,
      start_ns,
      end_ns,
      std::string()
    );
    activities_.push_back(row);
  }

  ~MockRoctracerLogger() {
    while (!activities_.empty()) {
      auto act = activities_.back();
      activities_.pop_back();
      free(act);
    }
  }

  std::vector<roctracerBase*> activities_;
  std::vector<std::pair<uint64_t, uint64_t>> externalCorrelations_[RoctracerLogger::CorrelationDomain::size];
};

// Mock parts of the RoctracerActivityApi
class MockRoctracerActivities : public RoctracerActivityApi {
 public:
  virtual int processActivities(
      std::function<void(const roctracerBase*)> handler,
      std::function<void(uint64_t, uint64_t, RoctracerLogger::CorrelationDomain)> correlationHandler) override {
    int count = 0;
    for (int it = RoctracerLogger::CorrelationDomain::begin; it < RoctracerLogger::CorrelationDomain::end; ++it) {
      auto &externalCorrelations = activityLogger->externalCorrelations_[it];
      for (auto &item : externalCorrelations) {
        correlationHandler(item.first, item.second, static_cast<RoctracerLogger::CorrelationDomain>(it));
      }
      externalCorrelations.clear();
    }
    for (auto &item : activityLogger->activities_) {
      handler(item);
      ++count;
    }
    return count;
  }

  std::unique_ptr<MockRoctracerLogger> activityLogger;
};

// Common setup / teardown and helper functions
class RoctracerActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<CuptiActivityProfiler>(
        roctracerActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
    loggerFactory.addProtocol("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<Config> cfg_;
  MockRoctracerActivities roctracerActivities_;
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

TEST_F(RoctracerActivityProfilerTest, SyncTrace) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(roctracerActivities_, /*cpu only*/ false);
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

  // And some CPU runtime ops, and GPU ops
  auto gpuOps = std::make_unique<MockRoctracerLogger>();
  gpuOps->addRuntimeKernelActivity(HIP_LAUNCH_KERNEL, start_time_ns + 33, start_time_ns + 38, 1);
  gpuOps->addRuntimeCopyActivity(HIP_MEMCPY, start_time_ns + 110, start_time_ns + 120, 2);
  gpuOps->addRuntimeKernelActivity(HIP_LAUNCH_KERNEL, start_time_ns + 130, start_time_ns + 145, 3);
  gpuOps->addRuntimeCopyActivity(HIP_MEMCPY, start_time_ns + 165, start_time_ns + 175, 4);
  gpuOps->addRuntimeKernelActivity(HIP_LAUNCH_KERNEL, start_time_ns + 195, start_time_ns + 205, 5);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  gpuOps->addMemcpyH2DActivity(start_time_ns + 140, start_time_ns + 150, 2);
  gpuOps->addKernelActivity(start_time_ns + 160, start_time_ns + 220, 3);
  gpuOps->addMemcpyD2HActivity(start_time_ns + 230, start_time_ns + 250, 4);
  gpuOps->addKernelActivity(start_time_ns + 260, start_time_ns + 280, 5);
  roctracerActivities_.activityLogger = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 15);
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
  // Check all activities are present and names are correct.
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["op4"], 1);
  EXPECT_EQ(activityCounts["op5"], 1);
  EXPECT_EQ(activityCounts["hipLaunchKernel"], 3);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Host -> Device)"], 1);
  EXPECT_EQ(activityCounts["Memcpy DtoH (Device -> Host)"], 1);
  EXPECT_EQ(activityCounts["kernel"], 3);

  auto sysTid = systemThreadId();
  // Check ops and runtime events are on thread sysTid
  EXPECT_EQ(resourceIds[sysTid], 10);
  // Kernels are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 3);
  EXPECT_EQ(resourceIds[2], 2);

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

TEST_F(RoctracerActivityProfilerTest, GpuNCCLCollectiveTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(roctracerActivities_, /*cpu only*/ false);
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
  auto gpuOps = std::make_unique<MockRoctracerLogger>();
  gpuOps->addCorrelationActivity(1, RoctracerLogger::CorrelationDomain::Domain0, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  roctracerActivities_.activityLogger = std::move(gpuOps);

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
  auto& cpu_op = trace.activities()->at(0);
  auto& gpu_kernel = trace.activities()->at(1);
  EXPECT_EQ(cpu_op->name(), kParamCommsCallName);
  EXPECT_EQ(gpu_kernel->name(), "kernel");

  // Check vector with length > 30 get truncated successfully
  std::vector<int64_t> expectedInSplit(kTruncatLength, 0);
  auto expectedInSplitStr =
      fmt::format("\"[{}, ...]\"", fmt::join(expectedInSplit, ", "));
  EXPECT_EQ(cpu_op->getMetadataValue(kInSplit), expectedInSplitStr);
  std::vector<int64_t> expectedGroupRanks(kTruncatLength - 1, 0);
  auto expectedGroupRanksStr = fmt::format(
      "\"[{}, ..., {}]\"", fmt::join(expectedGroupRanks, ", "), "0");
  EXPECT_EQ(
      cpu_op->getMetadataValue(kGroupRanks), expectedGroupRanksStr);

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

  // Check that the metadata fields are present in the JSON file
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

TEST_F(RoctracerActivityProfilerTest, GpuUserAnnotationTest) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(roctracerActivities_, /*cpu only*/ false);
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
  // RoctracerLogger::CorrelationDomain::Domain1 is used for user annotations.
  auto gpuOps = std::make_unique<MockRoctracerLogger>();
  gpuOps->addCorrelationActivity(1, RoctracerLogger::CorrelationDomain::Domain1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  gpuOps->addCorrelationActivity(1, RoctracerLogger::CorrelationDomain::Domain1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 15, kernelLaunchTime + 25, 1);
  roctracerActivities_.activityLogger = std::move(gpuOps);

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
  // Check that gpu_annotation covers both kernels
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

TEST_F(RoctracerActivityProfilerTest, SubActivityProfilers) {
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

  // Add a child profiler and check that it works
  MockRoctracerActivities activities;
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

TEST_F(RoctracerActivityProfilerTest, JsonGPUIDSortTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(roctracerActivities_, /*cpu only*/ false);
  int64_t start_time_ns = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));
  profiler.recordThreadInfo();

  // Set up CPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // Set up GPU events
  auto gpuOps = std::make_unique<MockRoctracerLogger>();
  gpuOps->addRuntimeKernelActivity(HIP_LAUNCH_KERNEL, start_time_ns + 23, start_time_ns + 28, 1);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  roctracerActivities_.activityLogger = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the contents of trace matches
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
      LOG(INFO) << sortLabel[event["pid"].asInt()];
    }
    if (event["name"] == "process_sort_index" && event["tid"] == 0 && event["pid"].isInt()) {
      sortIdx[event["pid"].asInt()] = event["args"]["sort_index"].asInt();
      LOG(INFO) << sortIdx[event["pid"].asInt()];
    }
  }

  // Expect atleast 16 GPU nodes, and 1 or more CPU nodes.
  EXPECT_LE(16, sortLabel.size());
  for (int i = 0; i<16; i++) {
    // Check there are 16 GPU sorts (0-15) with expected sort_index.
    EXPECT_EQ("GPU " + std::to_string(i), sortLabel[i]);
    // sortIndex is gpu + kExceedMaxPid to put GPU tracks at the bottom
    // of the trace timelines.
    EXPECT_EQ(i + kExceedMaxPid, sortIdx[i]);
  }
#endif
}
