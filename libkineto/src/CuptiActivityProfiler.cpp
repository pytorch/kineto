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

void CuptiActivityProfiler::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  std::lock_guard<std::mutex> guard(mutex_);
  const string& trace_name = cpuTrace->span.name;
  if (currentRunloopState_ != RunloopState::CollectTrace &&
      currentRunloopState_ != RunloopState::ProcessTrace) {
    VLOG(0) << "Trace collection not in progress - discarding span "
            << trace_name;
    return;
  }

  cpuTrace->span.iteration = iterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of span "
          << trace_name << " (" << cpuTrace->activities.size() << " activities / "
          << cpuTrace->gpuOpCount << " gpu activities)";
  traceBuffers_->cpu.push_back(std::move(cpuTrace));
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
    VLOG(0) << "Span time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    processCpuTrace(*cpu_trace, logger);
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
    ActivityLogger& logger) {
  if (cpuTrace.activities.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }

  CpuGpuSpanPair& span_pair = recordTraceSpan(cpuTrace.span, cpuTrace.gpuOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act.correlationId() << ": OP " << act.activityName;
    if (config_->selectedActivityTypes().count(act.type())) {
      act.log(logger);
    }
    clientActivityTraceMap_[act.correlationId()] = &span_pair;
    activityMap_[act.correlationId()] = &act;
  }
  logger.handleTraceSpan(cpu_span);
}

#ifdef HAS_CUPTI
inline void CuptiActivityProfiler::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
  cpuCorrelationMap_[correlation->correlationId] = correlation->externalId;
}
#endif // HAS_CUPTI

static GenericTraceActivity createUserGpuSpan(
    const libkineto::ITraceActivity& cpuTraceActivity,
    const libkineto::ITraceActivity& gpuTraceActivity) {
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
    const ITraceActivity&,
    const ITraceActivity& gpuActivity) {
  if (!gpuActivity.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
  const ITraceActivity& cpuActivity = *gpuActivity.linkedActivity();
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
inline bool CuptiActivityProfiler::outOfRange(const ITraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
        << " (" << act.timestamp() << " < " << captureWindowStartTime_ << " or "
        << (act.timestamp() + act.duration()) << " > " << captureWindowEndTime_;
  }
  return out_of_range;
}

void CuptiActivityProfiler::handleRuntimeActivity(
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
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked = linkedActivity(
      activity->correlationId, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
}

inline void CuptiActivityProfiler::updateGpuNetSpan(
    const ITraceActivity& gpuOp) {
  if (!gpuOp.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
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
void CuptiActivityProfiler::checkTimestampOrder(const ITraceActivity* act1) {
  // Correlated GPU runtime activity cannot
  // have timestamp greater than the GPU activity's
  const auto& it = correlatedCudaActivities_.find(act1->correlationId());
  if (it == correlatedCudaActivities_.end()) {
    correlatedCudaActivities_.insert({act1->correlationId(), act1});
    return;
  }

  // Activities may be appear in the buffers out of order.
  // If we have a runtime activity in the map, it should mean that we
  // have a GPU activity passed in, and vice versa.
  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::CUDA_RUNTIME) {
    // Buffer is out-of-order.
    // Swap so that runtime activity is first for the comparison below.
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    LOG(WARNING) << "GPU op timestamp (" << act2->timestamp()
                 << ") < runtime timestamp (" << act1->timestamp() << ") by "
                 << act1->timestamp() - act2->timestamp() << "us";
    LOG(WARNING) << "Name: " << act2->name()
                 << " Device: " << act2->deviceId()
                 << " Stream: " << act2->resourceId();
  }
}

inline void CuptiActivityProfiler::handleGpuActivity(
    const ITraceActivity& act,
    ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  VLOG(2) << act.correlationId() << ": "
          << act.name();
  recordStream(act.deviceId(), act.resourceId());
  act.log(*logger);
  updateGpuNetSpan(act);
}

const ITraceActivity* CuptiActivityProfiler::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    const auto& it2 = activityMap_.find(it->second);
    if (it2 != activityMap_.end()) {
      return it2->second;
    }
  }
  return nullptr;
}

template <class T>
inline void CuptiActivityProfiler::handleGpuActivity(
    const T* act, ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      act->correlationId, cpuCorrelationMap_);
  const auto& gpu_activity =
      traceBuffers_->addActivityWrapper(GpuActivity<T>(act, linked));
  handleGpuActivity(gpu_activity, logger);
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
            << config_->activitiesDuration().count() << " ms";
    auto session = profiler->configure(
        start_time_ms,
        config_->activitiesDuration().count(),
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

#if !USE_GOOGLE_LOG
  // Add a LoggerObserverCollector to collect all logs during the trace.
  loggerCollectorMetadata_ = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(loggerCollectorMetadata_.get());
#endif // !USE_GOOGLE_LOG

  config_ = config.clone();

  if (config_->activitiesDuration().count() == 0) {
    // Use default if not specified
    config_->setActivitiesDuration(
        config_->activitiesDurationDefault());
  }

  profileStartTime_ = config_->requestTimestamp();
  if (profileStartTime_ < now) {
    LOG(ERROR) << "Not starting tracing - start timestamp is in the past. Time difference (ms): " << duration_cast<milliseconds>(now - profileStartTime_).count();
    return;
  } else if ((profileStartTime_ - now) < config_->activitiesWarmupDuration()) {
    LOG(ERROR) << "Not starting tracing - insufficient time for warmup. Time to warmup (ms): " << duration_cast<milliseconds>(profileStartTime_ - now).count() ;
    return;
  }

  if (LOG_IS_ON(INFO)) {
    config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }
  if (!cpuOnly_ && !libkineto::api().client()) {
    LOG(INFO) << "GPU-only tracing for "
              << config_->activitiesDuration().count() << "ms";
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

  if (profilers_.size() > 0) {
    configureChildProfilers();
  }
  LOG(INFO) << "Tracing starting in "
            << duration_cast<seconds>(profileStartTime_ - now).count() << "s";

  traceBuffers_ = std::make_unique<ActivityBuffers>();
  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  currentRunloopState_ = RunloopState::Warmup;
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
            config_->activitiesDuration();
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
    for (const auto& it : iterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    iterationCountMap_.clear();
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

#if !USE_GOOGLE_LOG
  // Save logs from LoggerCollector objects into Trace metadata.
  auto LoggerMD = loggerCollectorMetadata_->extractCollectorMetadata();
  std::unordered_map<std::string, std::vector<std::string>> LoggerMDString;
  for (auto& md : LoggerMD) {
    LoggerMDString[toString(md.first)] = md.second;
  }
#endif // !USE_GOOGLE_LOG

  logger.finalizeTrace(config, std::move(traceBuffers_), captureWindowEndTime_, LoggerMDString);
}

void CuptiActivityProfiler::resetTraceData() {
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (!cpuOnly_) {
    cupti_.clearActivities();
  }
#endif // HAS_CUPTI || HAS_ROCTRACER
  activityMap_.clear();
  cpuCorrelationMap_.clear();
  correlatedCudaActivities_.clear();
  gpuUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
  sessions_.clear();
#if !USE_GOOGLE_LOG
  Logger::removeLoggerObserver(loggerCollectorMetadata_.get());
#endif // !USE_GOOGLE_LOG
}


} // namespace KINETO_NAMESPACE
