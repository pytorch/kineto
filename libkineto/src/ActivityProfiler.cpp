/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfiler.h"

#include <fmt/format.h>
#include <time.h>
#include <atomic>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

#ifdef HAS_CUPTI
#include <cupti.h>
#endif

#include "Config.h"
#include "time_since_epoch.h"
#ifdef HAS_CUPTI
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
#include "CuptiActivityInterface.h"
#endif // HAS_CUPTI
#include "output_base.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using namespace libkineto;
using std::string;

namespace KINETO_NAMESPACE {

bool ActivityProfiler::iterationTargetMatch(
    const libkineto::CpuTraceBuffer& trace) {
  const string& name = trace.span.name;
  bool match = (name == netIterationsTarget_);
  if (!match && applyNetFilterInternal(name) &&
      passesGpuOpCountThreshold(trace)) {
    if (netIterationsTarget_.empty()) {
      match = true;
      LOG(INFO) << "Target net for iterations not specified "
                << "- picking first encountered that passes net filter";
    } else if (name.find(netIterationsTarget_) != name.npos) {
      // Only track the first one that matches
      match = true;
    }
    if (match) {
      netIterationsTarget_ = name;
      LOG(INFO) << "Tracking net " << name << " for "
                << netIterationsTargetCount_ << " iterations";
    }
  }
  return match;
}

void ActivityProfiler::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  std::lock_guard<std::mutex> guard(mutex_);
  // FIXME: It's theoretically possible to receive a buffer from a
  // previous trace request. Probably should add a serial number.
  const string& trace_name = cpuTrace->span.name;
  if (currentRunloopState_ != RunloopState::CollectTrace &&
      currentRunloopState_ != RunloopState::ProcessTrace) {
    VLOG(0) << "Trace collection not in progress - discarding trace of net "
            << trace_name;
    return;
  }

  // Count iterations per net and stop profiling if the target net
  // has reached the iteration target (if no target net has been set,
  // one is picked at random)
  cpuTrace->span.iteration = netIterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of net "
          << trace_name << " (" << cpuTrace->activities.size() << " activities / "
          << cpuTrace->gpuOpCount << " gpu activities)";
  if (currentRunloopState_ == RunloopState::CollectTrace &&
      iterationTargetMatch(*cpuTrace)) {
    if (cpuTrace->span.iteration == 0) {
      VLOG(0) << "Setting profile start time from net to "
              << cpuTrace->span.startTime;
      captureWindowStartTime_ = cpuTrace->span.startTime;
    } else if (1 + cpuTrace->span.iteration >= netIterationsTargetCount_) {
      VLOG(0) << "Completed target iteration count for net "
              << trace_name;
      libkineto::api().client()->stop();
      // Tell the runloop to stop collection
      stopCollection_ = true;
      captureWindowEndTime_ = cpuTrace->span.endTime;
    }
  }

  traceBuffers_->cpu.push_back(std::move(cpuTrace));
}

bool ActivityProfiler::applyNetFilterInternal(const std::string& name) {
  if (netNameFilter_.empty()) {
    return true;
  }
  for (const std::string& match : netNameFilter_) {
    if (name.find(match) != name.npos) {
      return true;
    }
  }
  return false;
}

ActivityProfiler::ActivityProfiler(CuptiActivityInterface& cupti, bool cpuOnly)
    : cupti_(cupti),
      flushOverhead_{0, 0},
      setupOverhead_{0, 0},
      cpuOnly_{cpuOnly},
      currentRunloopState_{RunloopState::WaitForRequest},
      stopCollection_{false} {}

void ActivityProfiler::processTraceInternal(ActivityLogger& logger) {
  LOG(INFO) << "Processing " << traceBuffers_->cpu.size()
      << " CPU buffers";
  VLOG(0) << "Profile time range: " << captureWindowStartTime_ << " - "
          << captureWindowEndTime_;
  logger.handleTraceStart(metadata_);
  for (auto& cpu_trace : traceBuffers_->cpu) {
    string trace_name = cpu_trace->span.name;
    VLOG(0) << "Processing CPU buffer for " << trace_name << " ("
            << cpu_trace->span.iteration << ") - "
            << cpu_trace->activities.size() << " records";
    bool log_net = applyNetFilterInternal(trace_name) &&
        passesGpuOpCountThreshold(*cpu_trace) &&
        cpu_trace->span.startTime < captureWindowEndTime_ &&
        cpu_trace->span.endTime > captureWindowStartTime_;
    VLOG(0) << "Net time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    VLOG(0) << "Log net: " << (log_net ? "Yes" : "No");
    processCpuTrace(*cpu_trace, logger, log_net);
  }

#ifdef HAS_CUPTI
  if (!cpuOnly_) {
    VLOG(0) << "Retrieving GPU activity buffers";
    traceBuffers_->gpu = cupti_.activityBuffers();
    if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead_, cupti_.flushOverhead);
    }
    if (traceBuffers_->gpu) {
      const auto count_and_size = cupti_.processActivities(
          *traceBuffers_->gpu,
          std::bind(&ActivityProfiler::handleCuptiActivity, this, std::placeholders::_1, &logger));
      LOG(INFO) << "Processed " << count_and_size.first
                << " GPU records (" << count_and_size.second << " bytes)";
    }
  }
#endif // HAS_CUPTI

  finalizeTrace(*config_, logger);
}

ActivityProfiler::CpuGpuSpanPair& ActivityProfiler::recordTraceSpan(
    TraceSpan& span, int gpuOpCount) {
  TraceSpan gpu_span{
      0, 0, gpuOpCount, span.iteration, span.name, "GPU: "};
  auto& iterations = traceSpans_[span.name];
  iterations.push_back({span, gpu_span});
  return iterations.back();
}

void ActivityProfiler::processCpuTrace(
    libkineto::CpuTraceBuffer& cpuTrace,
    ActivityLogger& logger,
    bool logTrace) {
  if (cpuTrace.activities.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }

  CpuGpuSpanPair& span_pair = recordTraceSpan(cpuTrace.span, cpuTrace.gpuOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act.correlationId() << ": OP " << act.activityName;
    if (logTrace) {
      logger.handleCpuActivity(act, cpu_span);
    }
    // Stash event so we can look it up later when processing GPU trace
    externalEvents_.insertEvent(&act);
    clientActivityTraceMap_[act.correlationId()] = &span_pair;
  }
  if (logTrace) {
    logger.handleTraceSpan(cpu_span);
    if (cpu_span.name == netIterationsTarget_) {
      logger.handleIterationStart(cpu_span);
    }
  } else {
    disabledTraceSpans_.insert(cpu_span.name);
  }
}

#ifdef HAS_CUPTI
inline void ActivityProfiler::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
  switch(correlation->externalKind) {
    case CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0:
      externalEvents_.addCorrelation(
        correlation->externalId,
        correlation->correlationId,
        ExternalEventMap::CorrelationFlowType::Default);
      break;
    case CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1:
      externalEvents_.addCorrelation(
        correlation->externalId,
        correlation->correlationId,
        ExternalEventMap::CorrelationFlowType::User);
      break;
    default:
      LOG(ERROR) << "Received correlation activity with undefined kind: "
        << correlation->externalKind;
      break;
  }
  VLOG(2) << correlation->correlationId
          << ": CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION";
}
#endif // HAS_CUPTI

const libkineto::GenericTraceActivity&
ActivityProfiler::ExternalEventMap::getGenericTraceActivity(
  uint32_t id, CorrelationFlowType flowType) {
  static const libkineto::GenericTraceActivity nullOp_{};

  auto& correlationMap = getCorrelationMap(flowType);

  auto* res = events_[correlationMap[id]];
  if (res == nullptr) {
    // Entry may be missing because cpu trace hasn't been processed yet
    // Insert a dummy element so that we can check for this in insertEvent
    events_[correlationMap[id]] = &nullOp_;
    res = &nullOp_;
  }
  return *res;
}

void ActivityProfiler::ExternalEventMap::insertEvent(
    const libkineto::GenericTraceActivity* op) {
  if (events_[op->correlationId()] != nullptr) {
    LOG_EVERY_N(WARNING, 100)
        << "Events processed out of order - link will be missing";
  }
  events_[op->correlationId()] = op;
}

void ActivityProfiler::ExternalEventMap::addCorrelation(
  uint64_t external_id, uint32_t cuda_id, CorrelationFlowType flowType) {
  switch(flowType){
    case Default:
      defaultCorrelationMap_[cuda_id] = external_id;
      break;
    case User:
      userCorrelationMap_[cuda_id] = external_id;
      break;
  }
}

static void initUserGpuSpan(GenericTraceActivity& userTraceActivity,
  const libkineto::TraceActivity& cpuTraceActivity,
  const libkineto::TraceActivity& gpuTraceActivity) {
  userTraceActivity.device = gpuTraceActivity.deviceId();
  userTraceActivity.startTime = gpuTraceActivity.timestamp();
  userTraceActivity.sysThreadId = gpuTraceActivity.resourceId();
  userTraceActivity.endTime = gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  userTraceActivity.correlation = cpuTraceActivity.correlationId();
  userTraceActivity.activityType = cpuTraceActivity.type();
  userTraceActivity.activityName = cpuTraceActivity.name();
}

void ActivityProfiler::GpuUserEventMap::insertOrExtendEvent(
  const TraceActivity& cpuTraceActivity,
  const TraceActivity& gpuTraceActivity) {
  StreamKey key(gpuTraceActivity.deviceId(), gpuTraceActivity.resourceId());
  CorrelationSpanMap& correlationSpanMap = streamSpanMap[key];
  if (correlationSpanMap.count(cpuTraceActivity.correlationId()) == 0) {
    GenericTraceActivity& userTraceActivity = correlationSpanMap[cpuTraceActivity.correlationId()];
    initUserGpuSpan(userTraceActivity, cpuTraceActivity, gpuTraceActivity);
  }
  GenericTraceActivity& userTraceActivity = correlationSpanMap[cpuTraceActivity.correlationId()];
  if (gpuTraceActivity.timestamp() < userTraceActivity.startTime || userTraceActivity.startTime == 0) {
    userTraceActivity.startTime = gpuTraceActivity.timestamp();
  }
  if ((gpuTraceActivity.timestamp() + gpuTraceActivity.duration()) > userTraceActivity.endTime) {
    userTraceActivity.endTime = gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  }
}

void ActivityProfiler::GpuUserEventMap::logEvents(ActivityLogger *logger) {
  for (auto const& streamMapPair : streamSpanMap) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      logger->handleGenericActivity(
        correlationSpanPair.second);
    }
  }
}

#ifdef HAS_CUPTI
inline bool ActivityProfiler::outOfRange(const TraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
        << " (" << act.timestamp() << " < " << captureWindowStartTime_ << " or "
        << (act.timestamp() + act.duration()) << " > " << captureWindowEndTime_;
  }
  return out_of_range;
}

inline void ActivityProfiler::handleRuntimeActivity(
    const CUpti_ActivityAPI* activity,
    ActivityLogger* logger) {
  // Some CUDA calls that are very frequent and also not very interesting.
  // Filter these out to reduce trace size.
  if (activity->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020 ||
      activity->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020 ||
      activity->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020) {
    // Ignore these
    return;
  }
  VLOG(2) << activity->correlationId
          << ": CUPTI_ACTIVITY_KIND_RUNTIME, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  const GenericTraceActivity& ext =
    externalEvents_.getGenericTraceActivity(activity->correlationId,
      ExternalEventMap::CorrelationFlowType::Default);
  int32_t tid = activity->threadId;
  const auto& it = threadInfo_.find(tid);
  if (it != threadInfo_.end()) {
    tid = it->second.tid;
  }
  RuntimeActivity runtimeActivity(activity, ext, tid);
  if (ext.correlationId() == 0 && outOfRange(runtimeActivity)) {
    return;
  }
  if (!loggingDisabled(ext)) {
    runtimeActivity.log(*logger);
  }
}

inline void ActivityProfiler::updateGpuNetSpan(const TraceActivity& gpuOp) {
  const auto& it = clientActivityTraceMap_.find(
      gpuOp.linkedActivity()->correlationId());
  if (it == clientActivityTraceMap_.end()) {
    // No correlation id mapping?
    return;
  }
  TraceSpan& gpu_span = it->second->second;
  if (gpuOp.timestamp() < gpu_span.startTime || gpu_span.startTime == 0) {
    gpu_span.startTime = gpuOp.timestamp();
  }
  if ((gpuOp.timestamp() + gpuOp.duration()) > gpu_span.endTime) {
    gpu_span.endTime = gpuOp.timestamp() + gpuOp.duration();
  }
}

// I've observed occasional broken timestamps attached to GPU events...
static bool timestampsInCorrectOrder(
    const TraceActivity& ext,
    const TraceActivity& gpuOp) {
  if (ext.timestamp() > gpuOp.timestamp()) {
    LOG(WARNING) << "GPU op timestamp (" << gpuOp.timestamp()
                 << ") < runtime timestamp (" << ext.timestamp() << ") by "
                 << ext.timestamp() - gpuOp.timestamp() << "us";
    LOG(WARNING) << "Name: " << gpuOp.name()
                 << " Device: " << gpuOp.deviceId()
                 << " Stream: " << gpuOp.resourceId();
    return false;
  }
  return true;
}

inline void ActivityProfiler::handleGpuActivity(
    const TraceActivity& act,
    ActivityLogger* logger) {
  const TraceActivity& ext = *act.linkedActivity();
  if (ext.timestamp() == 0 && outOfRange(act)) {
    return;
  }
  // Correlated GPU runtime activity cannot have timestamp greater than the GPU activity's
  if (!timestampsInCorrectOrder(ext, act)) {
    return;
  }

  VLOG(2) << ext.correlationId() << "," << act.correlationId() << ": "
          << act.name();
  if (!loggingDisabled(ext)) {
    act.log(*logger);
    updateGpuNetSpan(act);
    const GenericTraceActivity& extUser =
      externalEvents_.getGenericTraceActivity(act.correlationId(),
        ExternalEventMap::CorrelationFlowType::User);
    // Correlated CPU activity cannot have timestamp greater than the GPU activity's
    if (!timestampsInCorrectOrder(extUser, act)) {
      return;
    }

    if (extUser.correlationId() != 0) {
      VLOG(2) << extUser.correlationId() << "," << act.correlationId()
              << " (user): "<< act.name();
      gpuUserEventMap_.insertOrExtendEvent(extUser, act);
    }
  }
}

template <class T>
inline void ActivityProfiler::handleGpuActivity(const T* act, ActivityLogger* logger) {
  const GenericTraceActivity& extDefault = externalEvents_.getGenericTraceActivity(act->correlationId,
      ExternalEventMap::CorrelationFlowType::Default);
  handleGpuActivity(GpuActivity<T>(act, extDefault), logger);
}

void ActivityProfiler::handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(
              record));
      break;
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const CUpti_ActivityAPI*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityKernel4*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpy*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpy2*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemset*>(record), logger);
      break;
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->kind;
      break;
  }
}
#endif // HAS_CUPTI

void ActivityProfiler::configure(
    const Config& config,
    const time_point<system_clock>& now) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (isActive()) {
    LOG(ERROR) << "ActivityProfiler already busy, terminating";
    return;
  }
  config_ = config.clone();

  if (config_->activitiesOnDemandDuration().count() == 0) {
    // Use default if not specified
    config_->setActivitiesOnDemandDuration(
        config_->activitiesOnDemandDurationDefault());
  }

  if (LOG_IS_ON(INFO)) {
    config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }
  if (!cpuOnly_ && !libkineto::api().client()) {
    LOG(INFO) << "GPU-only tracing for "
              << config_->activitiesOnDemandDuration().count() << "ms";
  } else {
    netNameFilter_ = config_->activitiesOnDemandExternalFilter();
    netGpuOpCountThreshold_ =
        config_->activitiesOnDemandExternalGpuOpCountThreshold();
    netIterationsTarget_ = config_->activitiesOnDemandExternalTarget();
    libkineto::api().setNetSizeThreshold(
        config_->activitiesOnDemandExternalNetSizeThreshold());
    netIterationsTargetCount_ = config_->activitiesOnDemandExternalIterations();

  }

  // Ensure we're starting in a clean state
  resetTraceData();

#ifdef HAS_CUPTI
  if (!cpuOnly_) {
    // Enabling CUPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling GPU tracing";
    cupti_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());

    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
    cupti_.enableCuptiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CUPTI

  profileStartTime_ = (config_->requestTimestamp() + config_->maxRequestAge()) +
      config_->activitiesWarmupDuration();
  if (profileStartTime_ < now) {
    profileStartTime_ = now + config_->activitiesWarmupDuration();
  }
  LOG(INFO) << "Tracing starting in "
            << duration_cast<seconds>(profileStartTime_ - now).count() << "s";

  traceBuffers_ = std::make_unique<ActivityBuffers>();
  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  currentRunloopState_ = RunloopState::Warmup;
}

void ActivityProfiler::startTraceInternal(const time_point<system_clock>& now) {
  captureWindowStartTime_ = libkineto::timeSinceEpoch(now);
  if (libkineto::api().client()) {
    libkineto::api().client()->start();
  }
  VLOG(0) << "Warmup -> CollectTrace";
  currentRunloopState_ = RunloopState::CollectTrace;
}

void ActivityProfiler::stopTraceInternal(const time_point<system_clock>& now) {
  if (captureWindowEndTime_ == 0) {
    captureWindowEndTime_ = libkineto::timeSinceEpoch(now);
  }
#ifdef HAS_CUPTI
  if (!cpuOnly_) {
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
    cupti_.disableCuptiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CUPTI
  if (currentRunloopState_ == RunloopState::CollectTrace) {
    VLOG(0) << "CollectTrace -> ProcessTrace";
  } else {
    LOG(WARNING) << "Called stopTrace with state == " <<
        static_cast<std::underlying_type<RunloopState>::type>(
            currentRunloopState_.load());
  }
  currentRunloopState_ = RunloopState::ProcessTrace;
}

void ActivityProfiler::resetInternal() {
  resetTraceData();
  currentRunloopState_ = RunloopState::WaitForRequest;
}

const time_point<system_clock> ActivityProfiler::performRunLoopStep(
    const time_point<system_clock>& now,
    const time_point<system_clock>& nextWakeupTime) {
  auto new_wakeup_time = nextWakeupTime;
  switch (currentRunloopState_) {
    case RunloopState::WaitForRequest:
      // Nothing to do
      break;

    case RunloopState::Warmup:
#ifdef HAS_CUPTI
      // Flushing can take a while so avoid doing it close to the start time
      if (!cpuOnly_ && nextWakeupTime < profileStartTime_) {
        cupti_.clearActivities();
      }

      if (cupti_.stopCollection) {
        // Go to process trace to clear any outstanding buffers etc
        if (libkineto::api().client()) {
          libkineto::api().client()->stop();
        }
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        resetInternal();
        VLOG(0) << "Warmup -> WaitForRequest";
        break;
      }
#endif // HAS_CUPTI

      if (now >= profileStartTime_) {
        if (now > profileStartTime_ + milliseconds(10)) {
          LOG(WARNING)
              << "Tracing started "
              << duration_cast<milliseconds>(now - profileStartTime_).count()
              << "ms late!";
        } else {
          LOG(INFO) << "Tracing started";
        }
        startTrace(now);
      } else if (nextWakeupTime > profileStartTime_) {
        new_wakeup_time = profileStartTime_;
      }

      break;

    case RunloopState::CollectTrace:
      // captureWindowStartTime_ can be set by external threads,
      // so recompute end time.
      // FIXME: Is this a good idea for synced start?
      {
        std::lock_guard<std::mutex> guard(mutex_);
        profileEndTime_ = time_point<system_clock>(
                              microseconds(captureWindowStartTime_)) +
            config_->activitiesOnDemandDuration();
      }

      if (now >= profileEndTime_ || stopCollection_.exchange(false)
#ifdef HAS_CUPTI
          || cupti_.stopCollection
#endif
      ){
        // Update runloop state first to prevent further updates to shared state
        LOG(INFO) << "Tracing complete";
        // FIXME: Need to communicate reason for stopping on errors
        if (libkineto::api().client()) {
          libkineto::api().client()->stop();
        }
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        VLOG_IF(0, now >= profileEndTime_) << "Reached profile end time";
      } else if (now < profileEndTime_ && profileEndTime_ < nextWakeupTime) {
        new_wakeup_time = profileEndTime_;
      }

      break;

    case RunloopState::ProcessTrace:
      // FIXME: Probably want to allow interruption here
      // for quickly handling trace request via synchronous API
      std::lock_guard<std::mutex> guard(mutex_);
      processTraceInternal(*logger_);
      resetInternal();
      VLOG(0) << "ProcessTrace -> WaitForRequest";
      break;
  }

  return new_wakeup_time;
}

void ActivityProfiler::finalizeTrace(const Config& config, ActivityLogger& logger) {
  LOG(INFO) << "Recorded nets:";
  {
    for (const auto& it : netIterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    netIterationCountMap_.clear();
  }

  // Process names
  string process_name = processName(processId());
  if (!process_name.empty()) {
    int32_t pid = processId();
    logger.handleProcessInfo(
        {pid, process_name, "CPU"}, captureWindowStartTime_);
    if (!cpuOnly_) {
      // GPU events use device id as pid (0-7).
      constexpr int kMaxGpuCount = 8;
      for (int gpu = 0; gpu < kMaxGpuCount; gpu++) {
        logger.handleProcessInfo(
            {gpu, process_name, fmt::format("GPU {}", gpu)},
            captureWindowStartTime_);
      }
    }
  }

  // Thread info
  for (auto pair : threadInfo_) {
    const auto& thread_info = pair.second;
    logger.handleThreadInfo(thread_info, captureWindowStartTime_);
  }

  for (const auto& iterations : traceSpans_) {
    for (const auto& span_pair : iterations.second) {
      const TraceSpan& gpu_span = span_pair.second;
      if (gpu_span.opCount > 0) {
        logger.handleTraceSpan(gpu_span);
      }
    }
  }

  gpuUserEventMap_.logEvents(&logger);

  logger.finalizeTrace(config, std::move(traceBuffers_), captureWindowEndTime_);
}

void ActivityProfiler::resetTraceData() {
#ifdef HAS_CUPTI
  if (!cpuOnly_) {
    cupti_.clearActivities();
  }
#endif // HAS_CUPTI
  externalEvents_.clear();
  gpuUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  disabledTraceSpans_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
}


} // namespace KINETO_NAMESPACE
