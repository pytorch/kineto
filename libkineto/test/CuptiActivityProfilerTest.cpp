/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <strings.h>
#include <time.h>
#include <chrono>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#endif

#include "include/libkineto.h"
#include "include/Config.h"
#include "src/ActivityProfiler.h"
#include "src/ConfigLoader.h"
#include "src/CuptiActivityApi.h"
#include "src/CuptiActivityProfiler.h"
#include "src/IConfigLoader.h"
#include "src/output_base.h"
#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"
#include "test/MockActivitySubProfiler.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

#define CUDA_LAUNCH_KERNEL CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
#define CUDA_MEMCPY CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020

struct MockTimeSinceEpoch {
  MOCK_METHOD0(nowUs, int64_t());
};

namespace {
const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}
}

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = TraceSpan(startTime, endTime,"Test trace");
    gpuOpCount = 0;
  }


  void addOp(std::string name, int64_t startTime, int64_t duration, int64_t correlation) {
    addOp(ActivityType::CPU_OP, name, startTime, duration, correlation);
  }

  void addOp(ActivityType type, std::string name, int64_t startTime, int64_t duration, int64_t correlation) {
    GenericTraceActivity op(span, type, name);
    op.startTime = startTime;
    op.endTime = startTime + duration;
    op.resource = systemThreadId();
    op.id = correlation;
    emplace_activity(std::move(op));
    span.opCount++;
  }
};

// Provides ability to easily create a few test CUPTI ops
struct MockCuptiActivityBuffer {
  void addCorrelationActivity(int64_t correlation, CUpti_ExternalCorrelationKind externalKind, int64_t externalId) {
    auto& act = *(CUpti_ActivityExternalCorrelation*) malloc(sizeof(CUpti_ActivityExternalCorrelation));
    act.kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
    act.externalId = externalId;
    act.externalKind = externalKind;
    act.correlationId = correlation;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addRuntimeActivity(
      CUpti_runtime_api_trace_cbid_enum cbid,
      int64_t start_us, int64_t duration_us, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityAPI>(
        start_us, start_us + duration_us, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
    act.cbid = cbid;
    act.threadId = threadId();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addKernelActivity(
      int64_t start_us, int64_t duration_us, int64_t correlation) {
    auto& act = createActivity<CUpti_ActivityKernel4>(
        start_us, start_us + duration_us, correlation);
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

// Mock parts of the CuptiActivityApi
class MockCuptiActivities : public CuptiActivityApi {
 public:
  MOCK_METHOD1(enableCuptiActivities, void(const std::set<ActivityType>& selected_activities));
  MOCK_METHOD0(disableCuptiActivities, void());
  MOCK_METHOD0(active, bool());

  const std::pair<int, int> processActivities(
      CuptiActivityBufferMap&, /*unused*/
      std::function<void(const CUpti_Activity*)> handler) override {
    for (CUpti_Activity* act : activityBuffer->activities) {
      handler(act);
    }
    return {activityBuffer->activities.size(), 100};
  }

  virtual std::unique_ptr<CuptiActivityBufferMap>
  activityBuffers() override {
    auto map = std::make_unique<CuptiActivityBufferMap>();
    auto buf = std::make_unique<CuptiActivityBuffer>(100);
    uint8_t* addr = buf->data();
    (*map)[addr] = std::move(buf);
    return map;
  }

  void bufferRequestedOverride(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    this->bufferRequested(buffer, size, maxNumRecords);
  }

  std::unique_ptr<MockCuptiActivityBuffer> activityBuffer;
};

class MockConfigLoader : public IConfigLoader {
 public:
  virtual ~MockConfigLoader() {}
  void addHandler(ConfigKind kind, IConfigHandler* handler) {
    addHandlerKind(kind);
    handler_ = handler;
  }
  MOCK_METHOD1(addHandlerKind, void(ConfigKind kind));
  MOCK_METHOD2(removeHandler, void(ConfigKind kind, IConfigHandler* handler));

  std::future<std::shared_ptr<IProfilerSession>>
  notify(const Config& cfg) {
    return handler_->acceptConfig(cfg);
  }

  IConfigHandler* handler_;
};

class MockCompositeProfilerSession : public ICompositeProfilerSession {
 public:
  MOCK_METHOD0(mutex, std::mutex&());
  MOCK_METHOD0(status, TraceStatus());
  MOCK_METHOD1(status, void(TraceStatus status));
  MOCK_METHOD0(errors, std::vector<std::string>());
  MOCK_METHOD1(log, void(ActivityLogger& logger));
  MOCK_METHOD1(
      registerCorrelationObserver, void(ICorrelationObserver* observer));
  MOCK_METHOD3(
      recordDeviceInfo,
      void(int64_t device, std::string name, std::string label));
  MOCK_CONST_METHOD1(deviceInfo, DeviceInfo(int64_t id));
  MOCK_METHOD0(recordThreadInfo, void());
  MOCK_METHOD4(recordResourceInfo,
      void(int64_t device, int64_t id, int sort_index, std::string name));
  MOCK_METHOD4(
      recordResourceInfo,
      void(
          int64_t device,
          int64_t id,
          int sort_index,
          std::function<std::string()> name));
  MOCK_CONST_METHOD2(
      resourceInfo, ResourceInfo(int64_t device, int64_t resource));
  MOCK_METHOD2(
      addMetadata, void(const std::string& key, const std::string& value));
  MOCK_METHOD2(
      addChild,
      void(
          const std::string& name,
          std::shared_ptr<IActivityProfilerSession> session));
  MOCK_CONST_METHOD1(
      session, IActivityProfilerSession*(const std::string& name));
  MOCK_METHOD0(startTime, int64_t());
  MOCK_METHOD0(endTime, int64_t());
  MOCK_METHOD0(activities, const std::vector<const ITraceActivity*>*());
  MOCK_METHOD1(save, void(const std::string& path));
  MOCK_METHOD2(pushCorrelationId, void(ActivityType kind, uint64_t id));
  MOCK_METHOD1(popCorrelationId, void(ActivityType kind));
};

// Common setup / teardown and helper functions
class CuptiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    startTime_ = time_point<system_clock>(microseconds(startTimeUs_));
    cpuOps_ = std::make_unique<MockCpuActivityBuffer>(
        startTimeUs_, startTimeUs_ + startTimeUs_);
    cpuOps_->addOp(ActivityType::USER_ANNOTATION, "user1", startTimeUs_ + 100, 200, 10);
    cpuOps_->addOp(ActivityType::USER_ANNOTATION, "user2", startTimeUs_ + 110, 50, 11);
    cpuOps_->addOp("op1", startTimeUs_ + 120, 30, 1);
    cpuOps_->addOp("op2", startTimeUs_ + 130, 10, 2);
    cpuOps_->addOp("op3", startTimeUs_ + 200, 50, 3);

    gpuOps_ = std::make_unique<MockCuptiActivityBuffer>();
    gpuOps_->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 1);
    // Cupti only adds the the bottom correlation record of a kind
    gpuOps_->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 11);
    gpuOps_->addRuntimeActivity(CUDA_LAUNCH_KERNEL, startTimeUs_ + 133, 5, 1);
    gpuOps_->addCorrelationActivity(2, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 2);
    gpuOps_->addCorrelationActivity(2, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 11);
    gpuOps_->addRuntimeActivity(CUDA_MEMCPY, startTimeUs_ + 210, 10, 2);
    gpuOps_->addCorrelationActivity(3, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 3);
    gpuOps_->addCorrelationActivity(3, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 10);
    gpuOps_->addRuntimeActivity(CUDA_LAUNCH_KERNEL, startTimeUs_ + 230, 15, 3);
    gpuOps_->addCorrelationActivity(4, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 4);
    gpuOps_->addCorrelationActivity(4, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 10);
    gpuOps_->addRuntimeActivity(CUDA_LAUNCH_KERNEL, startTimeUs_ + 250, 15, 4);
    gpuOps_->addKernelActivity(startTimeUs_ + 150, 20, 1);
    gpuOps_->addMemcpyActivity(startTimeUs_ + 240, 10, 2);
    gpuOps_->addKernelActivity(startTimeUs_ + 260, 60, 3);
    gpuOps_->addKernelActivity(startTimeUs_ + 350, 40, 4);

    bool success = cfg_.parse(fmt::format(R"CFG(
      ACTIVITIES_WARMUP_PERIOD_SECS = {}
      ACTIVITIES_DURATION_MSECS = {}
      PROFILE_START_TIME = {}
    )CFG", warmupSecs_, durationUs_ / 1000, startTimeUs_ / 1000));
    cfg_.validate(time_point<system_clock>(microseconds(0)));

    setMockTimeSinceEpoch([this](){
        return this->currentTime_.nowUs();
    });
  }

  static constexpr int warmupSecs_ = 1;
  static constexpr int startTimeUs_ = 1000000;
  time_point<system_clock> startTime_;
  static constexpr int durationUs_ = 40000;
  std::unique_ptr<MockCpuActivityBuffer> cpuOps_;
  std::unique_ptr<MockCuptiActivityBuffer> gpuOps_;
  MockCuptiActivities cuptiActivities_;
  ActivityLoggerFactory loggerFactory_;
  Config cfg_;
  MockTimeSinceEpoch currentTime_;
  MockConfigLoader configLoader_;
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
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}

TEST_F(CuptiActivityProfilerTest, SyncTrace) {
  using ::testing::Return;

  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"ActivityProfiler.cpp", "CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  CuptiActivityProfiler profiler("test", cuptiActivities_, /*cpu only*/ false);
  MockCompositeProfilerSession parent_session;
  EXPECT_CALL(currentTime_, nowUs())
      .Times(1)
      .WillOnce(Return(0));

  auto session = profiler.configure(cfg_, &parent_session);

  EXPECT_EQ(session->status(), TraceStatus::WARMUP);

  profiler.start(*session);

  EXPECT_EQ(session->status(), TraceStatus::RECORDING);

  profiler.stop(*session);

  EXPECT_EQ(session->status(), TraceStatus::PROCESSING);

  session->transferCpuTrace(std::move(cpuOps_));

  // And some GPU ops
  cuptiActivities_.activityBuffer = std::move(gpuOps_);

  // Get activities
  MemoryTraceLogger logger(cfg_);
  int32_t pid = processId();
  int32_t tid = threadId();
  EXPECT_CALL(parent_session, deviceInfo(pid))
      .Times(1)
      .WillOnce(Return(DeviceInfo(0, "gpu", "")));
  EXPECT_CALL(parent_session, startTime())
      .Times(9)
      .WillRepeatedly(Return(startTimeUs_));
  EXPECT_CALL(parent_session, endTime())
      .Times(9)
      .WillRepeatedly(Return(startTimeUs_ + durationUs_));

  session->log(logger);

  const auto activities = logger.traceActivities();
  EXPECT_EQ(activities->size(), 16);
  // name -> count
  std::map<std::string, int> activity_counts;
  // resourceID -> count
  std::map<int64_t, int> resource_ids;
  // (resource, id) -> activity
  std::map<std::pair<int64_t, int64_t>, const ITraceActivity*> activity_map;
  for (const auto* activity : *activities) {
    activity_counts[activity->name()]++;
    resource_ids[activity->resourceId()]++;
    activity_map[{activity->resourceId(), activity->correlationId()}] = activity;
  }
  for (const auto& p : activity_counts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  EXPECT_EQ(activity_counts["op1"], 1);
  EXPECT_EQ(activity_counts["op2"], 1);
  EXPECT_EQ(activity_counts["op3"], 1);
  EXPECT_EQ(activity_counts["cudaLaunchKernel"], 3);
  EXPECT_EQ(activity_counts["cudaMemcpy"], 1);
  EXPECT_EQ(activity_counts["kernel"], 3);
  EXPECT_EQ(activity_counts["Memcpy HtoD (Pinned -> Device)"], 1);

  int32_t systid = systemThreadId();
  // Ops and runtime events are on thread sysTid
  EXPECT_EQ(resource_ids[systid], 8);
  // Kernels are on stream 1, memcpy on stream 2
  // Stream 1: 3 kernels + 2 user annotations
  EXPECT_EQ(resource_ids[1], 5);
  // Stream 2: 1 memcpy + 1 user annotation
  EXPECT_EQ(resource_ids[2], 2);

  // User annotations
  // We should have additional annotation activities created
  // on the GPU timeline.
  EXPECT_EQ(activity_counts["user1"], 2);
  EXPECT_EQ(activity_counts["user2"], 3);

  auto& user1 = activity_map[{systid, 10}];
  auto& user2 = activity_map[{systid, 11}];
  auto& kernel1 = activity_map[{1, 1}];
  auto& memcpy = activity_map[{2, 2}];
  auto& kernel2 = activity_map[{1, 3}];
  auto& kernel3 = activity_map[{1, 4}];
  auto& user_gpu1 = activity_map[{1, 10}];
  auto& user_gpu2_1 = activity_map[{1, 11}];
  auto& user_gpu2_2 = activity_map[{2, 11}];
  EXPECT_EQ(user_gpu1->type(), ActivityType::GPU_USER_ANNOTATION);
  EXPECT_EQ(user_gpu2_1->type(), ActivityType::GPU_USER_ANNOTATION);
  EXPECT_EQ(user_gpu2_2->type(), ActivityType::GPU_USER_ANNOTATION);
  EXPECT_EQ(user_gpu1->timestamp(), kernel2->timestamp());
  EXPECT_EQ(user_gpu2_1->timestamp(), kernel1->timestamp());
  EXPECT_EQ(user_gpu2_2->timestamp(), memcpy->timestamp());
  EXPECT_EQ(user_gpu1->deviceId(), kernel2->deviceId());
  EXPECT_EQ(user_gpu1->resourceId(), kernel2->resourceId());
  EXPECT_EQ(user_gpu1->correlationId(), user1->correlationId());
  EXPECT_EQ(user_gpu1->name(), user1->name());
  EXPECT_EQ(
      user_gpu1->duration(),
      kernel3->timestamp() + kernel3->duration() - kernel2->timestamp());
  EXPECT_EQ(user_gpu2_1->duration(), kernel1->duration());
  EXPECT_EQ(user_gpu2_2->duration(), memcpy->duration());

#ifdef __linux__
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  ChromeTraceLogger file_logger(filename);
  logger.log(file_logger);
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  LOG(INFO) << "Size: " << buf.st_size;
  EXPECT_GT(buf.st_size, 100);
#endif
}

TEST_F(CuptiActivityProfilerTest, CompositeAsyncTrace) {
  using ::testing::Return;

  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  EXPECT_CALL(configLoader_, addHandlerKind(ConfigLoader::ConfigKind::ActivityProfiler))
    .Times(1);
  ActivityProfiler profiler("Test", configLoader_);
  const std::string cupti_profiler_name = "Cupti Test Profiler";
  ActivityProfilerFactory::Creator creator =
      [&cupti_profiler_name, this]() -> std::unique_ptr<IActivityProfiler> {
        return std::make_unique<CuptiActivityProfiler>(
            cupti_profiler_name, this->cuptiActivities_, /*cpu only*/ false);};
  profiler.registerProfiler(cupti_profiler_name, creator);
  profiler.registerLogger("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
  });
  profiler.init(nullptr);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);

  int64_t start_time_ms = 1000;
  int64_t warmup = 1;

  bool success = cfg_.parse(fmt::format(R"CFG(
    ACTIVITIES_LOG_FILE = {}
  )CFG", filename));

  EXPECT_TRUE(success);

  // Materialize some activities
  cuptiActivities_.activityBuffer = std::move(gpuOps_);

  EXPECT_CALL(currentTime_, nowUs()).Times(3)
      .WillOnce(Return(0))
      .WillOnce(Return(startTimeUs_))
      .WillOnce(Return(startTimeUs_ + durationUs_));
  auto future = configLoader_.notify(cfg_);
  auto s = future.get();
  EXPECT_TRUE(s != nullptr);
  auto& session = dynamic_cast<IActivityProfilerSession&>(*s);

  EXPECT_EQ(session.status(), TraceStatus::PROCESSING);

#ifdef __linux__
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

  char* contents = (char*) malloc(buf.st_size);
  read(fd, contents, buf.st_size);
  EXPECT_TRUE(strstr(contents, "150") != nullptr);

  free(contents);
  close(fd);
#endif
}

TEST_F(CuptiActivityProfilerTest, BufferSizeLimitTestWarmup) {
  using ::testing::Return;

  int maxBufferSizeMB = 3;
  bool success = cfg_.parse(fmt::format(R"CFG(
    ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB = {}
    ACTIVITY_TYPES = kernel
  )CFG", maxBufferSizeMB));

  EXPECT_TRUE(success);

  CuptiActivityProfiler profiler("test", cuptiActivities_, /*cpu only*/ false);
  MockCompositeProfilerSession parent_session;
  EXPECT_CALL(currentTime_, nowUs())
      .Times(1)
      .WillOnce(Return(0));
  std::set<ActivityType> activities = {ActivityType::CONCURRENT_KERNEL};
  EXPECT_CALL(cuptiActivities_, enableCuptiActivities(activities)).Times(1);

  auto session = profiler.configure(cfg_, &parent_session);

  EXPECT_EQ(session->status(), TraceStatus::WARMUP);
  EXPECT_FALSE(cuptiActivities_.error());

  // Make enough buffer requests to exceed limit.
  // Cupti activity profiling should be disabled as a result.
  EXPECT_CALL(cuptiActivities_, disableCuptiActivities());
  for (size_t i = 0; i < maxBufferSizeMB; i++) {
    uint8_t* buf;
    size_t gpuBufferSize;
    size_t maxNumRecords;
    cuptiActivities_.bufferRequestedOverride(&buf, &gpuBufferSize, &maxNumRecords);
  }

  EXPECT_FALSE(cuptiActivities_.active());
  EXPECT_TRUE(cuptiActivities_.error());

  // Attempting to start profiler will now result in an error
  profiler.start(*session);
  EXPECT_EQ(session->status(), TraceStatus::ERROR);
}

