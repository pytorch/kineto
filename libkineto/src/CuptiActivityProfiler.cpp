/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivityProfiler.h"

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
#include "CuptiActivityApi.h"
#endif // HAS_CUPTI
#ifdef HAS_ROCTRACER
#include "RoctracerActivityApi.h"
#endif
#include "output_base.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using namespace libkineto;
using std::string;

namespace KINETO_NAMESPACE {

bool CuptiActivityProfiler::iterationTargetMatch(
    libkineto::CpuTraceBuffer& trace) {
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
      trace.span.tracked = true;
      LOG(INFO) << "Tracking net " << name << " for "
                << netIterationsTargetCount_ << " iterations";
    }
  }
  return match;
}

void CuptiActivityProfiler::transferCpuTrace(
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

bool CuptiActivityProfiler::applyNetFilterInternal(const std::string& name) {
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

#ifdef HAS_ROCTRACER
CuptiActivityProfiler::CuptiActivityProfiler(RoctracerActivityApi& cupti, bool cpuOnly)
#else
CuptiActivityProfiler::CuptiActivityProfiler(CuptiActivityApi& cupti, bool cpuOnly)
#endif
    : cupti_(cupti),
      flushOverhead_{0, 0},
      setupOverhead_{0, 0},
      cpuOnly_{cpuOnly},
      currentRunloopState_{RunloopState::WaitForRequest},
      stopCollection_{false} {}

void CuptiActivityProfiler::processTraceInternal(ActivityLogger& logger) {
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
          std::bind(&CuptiActivityProfiler::handleCuptiActivity, this, std::placeholders::_1, &logger));
      LOG(INFO) << "Processed " << count_and_size.first
                << " GPU records (" << count_and_size.second << " bytes)";
    }
  }
#endif // HAS_CUPTI
#ifdef HAS_ROCTRACER
  if (!cpuOnly_) {
    VLOG(0) << "Retrieving GPU activity buffers";
    const int count = cupti_.processActivities(logger);
    LOG(INFO) << "Processed " << count
              << " GPU records";
  }
#endif // HAS_ROCTRACER

  for (const auto& session : sessions_){
    LOG(INFO) << "Processing child profiler trace";
    session->processTrace(logger);
  }

  finalizeTrace(*config_, logger);
}

CuptiActivityProfiler::CpuGpuSpanPair& CuptiActivityProfiler::recordTraceSpan(
    TraceSpan& span, int gpuOpCount) {
  TraceSpan gpu_span(gpuOpCount, span.iteration, span.name, "GPU: ");
  auto& iterations = traceSpans_[span.name];
  iterations.push_back({span, gpu_span});
  return iterations.back();
}

void CuptiActivityProfiler::processCpuTrace(
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
    if (logTrace && config_->selectedActivityTypes().count(act.type())) {
      act.log(logger);
    }
    // Stash event so we can look it up later when processing GPU trace
    externalEvents_.insertEvent(&act);
    clientActivityTraceMap_[act.correlationId()] = &span_pair;
  }
  if (logTrace) {
    logger.handleTraceSpan(cpu_span);
  } else {
    disabledTraceSpans_.insert(cpu_span.name);
  }
}

#ifdef HAS_CUPTI
inline void CuptiActivityProfiler::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
    externalEvents_.addCorrelation(
        correlation->externalId, correlation->correlationId);
}
#endif // HAS_CUPTI

const libkineto::GenericTraceActivity&
CuptiActivityProfiler::ExternalEventMap::correlatedActivity(uint32_t id) {
  static const libkineto::GenericTraceActivity nullOp_(
      defaultTraceSpan().first, ActivityType::CPU_OP, "NULL");

  auto* res = events_[correlationMap_[id]];
  if (res == nullptr) {
    // Entry may be missing because cpu trace hasn't been processed yet
    // Insert a dummy element so that we can check for this in insertEvent
    events_[correlationMap_[id]] = &nullOp_;
    res = &nullOp_;
  }
  return *res;
}

void CuptiActivityProfiler::ExternalEventMap::insertEvent(
    const libkineto::GenericTraceActivity* op) {
  if (events_[op->correlationId()] != nullptr) {
    LOG_EVERY_N(WARNING, 100)
        << "Events processed out of order - link will be missing";
  }
  events_[op->correlationId()] = op;
}

void CuptiActivityProfiler::ExternalEventMap::addCorrelation(
    uint64_t external_id, uint32_t cuda_id) {
  correlationMap_[cuda_id] = external_id;
}

static GenericTraceActivity createUserGpuSpan(
    const libkineto::TraceActivity& cpuTraceActivity,
    const libkineto::TraceActivity& gpuTraceActivity) {
  GenericTraceActivity res(
      *cpuTraceActivity.traceSpan(),
      ActivityType::GPU_USER_ANNOTATION,
      cpuTraceActivity.name());
  res.startTime = gpuTraceActivity.timestamp();
  res.device = gpuTraceActivity.deviceId();
  res.resource = gpuTraceActivity.resourceId();
  res.endTime =
      gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void CuptiActivityProfiler::GpuUserEventMap::insertOrExtendEvent(
    const TraceActivity&,
    const TraceActivity& gpuActivity) {
  const TraceActivity& cpuActivity = *gpuActivity.linkedActivity();
  StreamKey key(gpuActivity.deviceId(), gpuActivity.resourceId());
  CorrelationSpanMap& correlationSpanMap = streamSpanMap_[key];
  auto it = correlationSpanMap.find(cpuActivity.correlationId());
  if (it == correlationSpanMap.end()) {
    auto it_success = correlationSpanMap.insert({
        cpuActivity.correlationId(), createUserGpuSpan(cpuActivity, gpuActivity)
    });
    it = it_success.first;
  }
  GenericTraceActivity& span = it->second;
  if (gpuActivity.timestamp() < span.startTime || span.startTime == 0) {
    span.startTime = gpuActivity.timestamp();
  }
  int64_t gpu_activity_end = gpuActivity.timestamp() + gpuActivity.duration();
  if (gpu_activity_end > span.endTime) {
    span.endTime = gpu_activity_end;
  }
}

const CuptiActivityProfiler::CpuGpuSpanPair& CuptiActivityProfiler::defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  static CpuGpuSpanPair span_pair(span, span);
  return span_pair;
}

void CuptiActivityProfiler::GpuUserEventMap::logEvents(ActivityLogger *logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

#ifdef HAS_CUPTI
inline bool CuptiActivityProfiler::outOfRange(const TraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
        << " (" << act.timestamp() << " < " << captureWindowStartTime_ << " or "
        << (act.timestamp() + act.duration()) << " > " << captureWindowEndTime_;
  }
  return out_of_range;
}

inline void CuptiActivityProfiler::handleRuntimeActivity(
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
      externalEvents_.correlatedActivity(activity->correlationId);
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  RuntimeActivity runtimeActivity(activity, ext, tid);
  if (ext.correlationId() == 0 && outOfRange(runtimeActivity)) {
    return;
  }
  if (!loggingDisabled(ext)) {
    runtimeActivity.log(*logger);
  }
}

inline void CuptiActivityProfiler::updateGpuNetSpan(const TraceActivity& gpuOp) {
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

inline void CuptiActivityProfiler::handleGpuActivity(
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
    recordStream(act.deviceId(), act.resourceId());
    act.log(*logger);
    updateGpuNetSpan(act);
    /*
    const GenericTraceActivity& extUser =
        externalEvents_.correlatedActivity(act.correlationId());
    // Correlated CPU activity cannot have timestamp greater than the GPU activity's
    if (!timestampsInCorrectOrder(extUser, act)) {
      return;
    }
    if (extUser.correlationId() != 0) {
      VLOG(2) << extUser.correlationId() << "," << act.correlationId()
              << " (user): "<< act.name();
*/
    if (config_->selectedActivityTypes().count(ActivityType::GPU_USER_ANNOTATION) &&
        act.linkedActivity() &&
        act.linkedActivity()->type() == ActivityType::USER_ANNOTATION) {
      //gpuUserEventMap_.insertOrExtendEvent(act, act);
    }
//    }
  }
}

template <class T>
inline void CuptiActivityProfiler::handleGpuActivity(
    const T* act, ActivityLogger* logger) {
  const GenericTraceActivity& extDefault =
      externalEvents_.correlatedActivity(act->correlationId);
  handleGpuActivity(GpuActivity<T>(act, extDefault), logger);
}

void CuptiActivityProfiler::handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger) {
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

void CuptiActivityProfiler::configureChildProfilers() {
  // If child profilers are enabled create profiler sessions
  for (auto& profiler: profilers_) {
    int64_t start_time_ms = duration_cast<milliseconds>(
        profileStartTime_.time_since_epoch()).count();
    LOG(INFO) << "Running child profiler " << profiler->name() << " for "
            << config_->activitiesOnDemandDuration().count() << " ms";
    auto session = profiler->configure(
        start_time_ms,
        config_->activitiesOnDemandDuration().count(),
        config_->selectedActivityTypes()
    );
    if (session) {
      sessions_.push_back(std::move(session));
    }
  }
}

void CuptiActivityProfiler::configure(
    const Config& config,
    const time_point<system_clock>& now) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (isActive()) {
    LOG(ERROR) << "CuptiActivityProfiler already busy, terminating";
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

#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
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
#ifdef HAS_CUPTI
    cupti_.enableCuptiActivities(config_->selectedActivityTypes());
#else
    cupti_.enableActivities(config_->selectedActivityTypes());
#endif
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CUPTI || HAS_ROCTRACER

  profileStartTime_ = config_->requestTimestamp();

  if (profileStartTime_ < now) {
    LOG(ERROR) << "Not starting tracing - start timestamp is in the past. Time difference (ms): " << duration_cast<milliseconds>(now - profileStartTime_).count();
  } else if ((profileStartTime_ - now) < config_->activitiesWarmupDuration()) {
    LOG(ERROR) << "Not starting tracing - insufficient time for warmup. Time to warmup (ms): " << duration_cast<milliseconds>(profileStartTime_ - now).count() ;
  } else {
    if (profilers_.size() > 0) {
      configureChildProfilers();
    }
    LOG(INFO) << "Tracing starting in "
              << duration_cast<seconds>(profileStartTime_ - now).count() << "s";

    traceBuffers_ = std::make_unique<ActivityBuffers>();
    captureWindowStartTime_ = captureWindowEndTime_ = 0;
    currentRunloopState_ = RunloopState::Warmup;
  }
}

void CuptiActivityProfiler::startTraceInternal(const time_point<system_clock>& now) {
  captureWindowStartTime_ = libkineto::timeSinceEpoch(now);
  VLOG(0) << "Warmup -> CollectTrace";
  for (auto& session: sessions_){
    LOG(INFO) << "Starting child profiler session";
    session->start();
  }
  currentRunloopState_ = RunloopState::CollectTrace;
}

void CuptiActivityProfiler::stopTraceInternal(const time_point<system_clock>& now) {
  if (captureWindowEndTime_ == 0) {
    captureWindowEndTime_ = libkineto::timeSinceEpoch(now);
  }
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (!cpuOnly_) {
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
#ifdef HAS_CUPTI
    cupti_.disableCuptiActivities(config_->selectedActivityTypes());
#else
    cupti_.disableActivities(config_->selectedActivityTypes());
#endif
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CUPTI || HAS_ROCTRACER

  if (currentRunloopState_ == RunloopState::CollectTrace) {
    VLOG(0) << "CollectTrace -> ProcessTrace";
  } else {
    LOG(WARNING) << "Called stopTrace with state == " <<
        static_cast<std::underlying_type<RunloopState>::type>(
            currentRunloopState_.load());
  }
  for (auto& session: sessions_){
    LOG(INFO) << "Stopping child profiler session";
    session->stop();
  }
  currentRunloopState_ = RunloopState::ProcessTrace;
}

void CuptiActivityProfiler::resetInternal() {
  resetTraceData();
  currentRunloopState_ = RunloopState::WaitForRequest;
}

const time_point<system_clock> CuptiActivityProfiler::performRunLoopStep(
    const time_point<system_clock>& now,
    const time_point<system_clock>& nextWakeupTime) {
  auto new_wakeup_time = nextWakeupTime;
  switch (currentRunloopState_) {
    case RunloopState::WaitForRequest:
      // Nothing to do
      break;

    case RunloopState::Warmup:
      VLOG(1) << "State: Warmup";
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
      // Flushing can take a while so avoid doing it close to the start time
      if (!cpuOnly_ && nextWakeupTime < profileStartTime_) {
        cupti_.clearActivities();
      }

      if (cupti_.stopCollection) {
        // Go to process trace to clear any outstanding buffers etc
        LOG(WARNING) << "Trace terminated during warmup";
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        resetInternal();
        VLOG(0) << "Warmup -> WaitForRequest";
        break;
      }
#endif // HAS_CUPTI || HAS_ROCTRACER

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
        if (libkineto::api().client()) {
          libkineto::api().client()->start();
        }
        if (nextWakeupTime > profileEndTime_) {
          new_wakeup_time = profileEndTime_;
        }
      } else if (nextWakeupTime > profileStartTime_) {
        new_wakeup_time = profileStartTime_;
      }

      break;

    case RunloopState::CollectTrace:
      VLOG(1) << "State: CollectTrace";
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
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
          || cupti_.stopCollection
#endif // HAS_CUPTI || HAS_ROCTRACER
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
      VLOG(1) << "State: ProcessTrace";
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

void CuptiActivityProfiler::finalizeTrace(const Config& config, ActivityLogger& logger) {
  LOG(INFO) << "Recorded nets:";
  {
    for (const auto& it : netIterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    netIterationCountMap_.clear();
  }

  // Process names
  int32_t pid = processId();
  string process_name = processName(pid);
  if (!process_name.empty()) {
    logger.handleDeviceInfo(
        {pid, process_name, "CPU"}, captureWindowStartTime_);
    if (!cpuOnly_) {
      // GPU events use device id as pid (0-7).
      constexpr int kMaxGpuCount = 8;
      for (int gpu = 0; gpu < kMaxGpuCount; gpu++) {
        logger.handleDeviceInfo(
            {gpu, process_name, fmt::format("GPU {}", gpu)},
            captureWindowStartTime_);
      }
    }
  }

  // Thread & stream info
  for (auto pair : resourceInfo_) {
    const auto& resource = pair.second;
    logger.handleResourceInfo(resource, captureWindowStartTime_);
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

void CuptiActivityProfiler::resetTraceData() {
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (!cpuOnly_) {
    cupti_.clearActivities();
  }
#endif // HAS_CUPTI || HAS_ROCTRACER
  externalEvents_.clear();
  gpuUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  disabledTraceSpans_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
  sessions_.clear();
}


} // namespace KINETO_NAMESPACE
