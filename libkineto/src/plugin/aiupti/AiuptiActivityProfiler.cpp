#include "AiuptiActivityProfiler.h"
#include "AiuptiActivityApi.h"

#include <chrono>

namespace KINETO_NAMESPACE {

uint32_t AiuptiActivityProfilerSession::iterationCount_ = 0;
std::vector<std::array<unsigned char, 16>>
    AiuptiActivityProfilerSession::deviceUUIDs_ = {};
std::vector<std::string> AiuptiActivityProfilerSession::correlateRuntimeOps_ = {
    "launchCb",
    "launchComputeStream"};

// =========== Session Constructor ============= //
AiuptiActivityProfilerSession::AiuptiActivityProfilerSession(
    AiuptiActivityApi& api,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : api_(api), config_(config.clone()), activity_types_(activity_types) {
  api_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());
}

AiuptiActivityProfilerSession::~AiuptiActivityProfilerSession() {
  api_.clearActivities();
}

// =========== Session Public Methods ============= //
void AiuptiActivityProfilerSession::start() {
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  api_.enableAiuptiActivities(activity_types_);
}

void AiuptiActivityProfilerSession::stop() {
  profilerEndTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  api_.clearActivities();
  api_.disablePtiActivities(activity_types_);
}

void AiuptiActivityProfilerSession::processTrace(ActivityLogger& logger) {
  traceBuffer_.span = libkineto::TraceSpan(
      profilerStartTs_, profilerEndTs_, "__aiu_profiler__");
  traceBuffer_.span.iteration = iterationCount_++;

  auto aiuBuffer = api_.activityBuffers();
  if (aiuBuffer) {
    api_.processActivities(
        *aiuBuffer,
        std::bind(
            &AiuptiActivityProfilerSession::handlePtiActivity,
            this,
            std::placeholders::_1,
            &logger));
  }
}

void AiuptiActivityProfilerSession::processTrace(
    ActivityLogger& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

// TODO (mcalman): support multi-AIU
std::unique_ptr<libkineto::DeviceInfo>
AiuptiActivityProfilerSession::getDeviceInfo() {
  int32_t pid = processId();
  std::string process_name = processName(pid);
  int aiu = 0;
  return std::make_unique<DeviceInfo>(
      aiu, aiu + kExceedMaxPid, process_name, fmt::format("AIU {}", 0));
}

std::vector<libkineto::ResourceInfo>
AiuptiActivityProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> resourceInfos;
  for (const auto& entries : resourceInfo_) {
    resourceInfos.push_back(entries.second);
  }
  return resourceInfos;
}

bool AiuptiActivityProfilerSession::hasDeviceResource(
    uint32_t device,
    uint32_t id) {
  return resourceInfo_.find({device, id}) != resourceInfo_.end();
}

void AiuptiActivityProfilerSession::recordStream(uint32_t device, uint32_t id) {
  if (!hasDeviceResource(device, id)) {
    resourceInfo_.emplace(
        std::make_pair(device, id),
        ResourceInfo(
            device, id, kExceedMaxTid + id, fmt::format("Stream {}", id)));
  }
}

void AiuptiActivityProfilerSession::recordMemoryStream(
    uint32_t device,
    uint32_t id,
    std::string name) {
  if (!hasDeviceResource(device, id)) {
    resourceInfo_.emplace(
        std::make_pair(device, id), ResourceInfo(device, id, id, name));
  }
}

std::unique_ptr<libkineto::CpuTraceBuffer>
AiuptiActivityProfilerSession::getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

void AiuptiActivityProfilerSession::pushCorrelationId(uint64_t id) {
  api_.pushCorrelationID(id, AiuptiActivityApi::CorrelationFlowType::Default);
}

void AiuptiActivityProfilerSession::popCorrelationId() {
  api_.popCorrelationID(AiuptiActivityApi::CorrelationFlowType::Default);
}

void AiuptiActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  api_.pushCorrelationID(id, AiuptiActivityApi::CorrelationFlowType::User);
}

void AiuptiActivityProfilerSession::popUserCorrelationId() {
  api_.popCorrelationID(AiuptiActivityApi::CorrelationFlowType::User);
}

// =========== ActivityProfiler Public Methods ============= //
const std::set<ActivityType> kAiuTypes{
    ActivityType::GPU_MEMCPY,
    ActivityType::GPU_MEMSET,
    ActivityType::CONCURRENT_KERNEL,
    ActivityType::PRIVATEUSE1_RUNTIME,
};

const std::string& AIUActivityProfiler::name() const {
  return name_;
}

const std::set<ActivityType>& AIUActivityProfiler::availableActivities() const {
  throw std::runtime_error(
      "The availableActivities is legacy method and should not be called by kineto");
  return kAiuTypes;
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AIUActivityProfiler::configure(
    const std::set<ActivityType>& activity_types,
    const libkineto::Config& config) {
  return std::make_unique<AiuptiActivityProfilerSession>(
      AiuptiActivityApi::singleton(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AIUActivityProfiler::configure(
    int64_t ts_ms,
    int64_t duration_ms,
    const std::set<ActivityType>& activity_types,
    const libkineto::Config& config) {
  AsyncProfileStartTime_ = ts_ms;
  AsyncProfileEndTime_ = ts_ms + duration_ms;
  return configure(activity_types, config);
}
} // namespace KINETO_NAMESPACE
