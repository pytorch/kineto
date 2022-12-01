/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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
#include <type_traits>
#include <vector>
#include <limits>

#ifdef HAS_CUPTI
#include <cupti.h>
// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "cuda_call.h"
#include "cupti_call.h"
#include "CudaUtil.h"
#endif

#include "Config.h"
#include "time_since_epoch.h"
#ifdef HAS_CUPTI
#include "CuptiActivity.h"
#include "CuptiActivity.cpp"
#include "CuptiActivityApi.h"
#endif // HAS_CUPTI
#ifdef HAS_ROCTRACER
#include "RoctracerActivityApi.h"
#endif
#include "output_base.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using std::string;

namespace KINETO_NAMESPACE {

const std::set<ActivityType>&
CuptiActivityProfiler::supportedActivityTypes() const {
  static const std::set<ActivityType> types = {
      ActivityType::CPU_OP,
      ActivityType::USER_ANNOTATION,
      ActivityType::CPU_INSTANT_EVENT,
      ActivityType::GPU_MEMCPY,
      ActivityType::GPU_MEMSET,
      ActivityType::CONCURRENT_KERNEL,
      ActivityType::EXTERNAL_CORRELATION,
      ActivityType::CUDA_RUNTIME,
      ActivityType::GPU_USER_ANNOTATION,
      ActivityType::GLOW_RUNTIME
  };
  return types;
}

void CuptiActivityProfilerSession::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  const string& trace_name = cpuTrace->span.name;
  if (status_ != TraceStatus::RECORDING &&
      status_ != TraceStatus::PROCESSING) {
    VLOG(0) << "Trace collection not in progress - discarding span "
            << trace_name;
    return;
  }

  cpuTrace->span.iteration = iterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of span "
          << trace_name << " (" << cpuTrace->activities.size() << " activities / "
          << cpuTrace->gpuOpCount << " gpu activities)";
  LOG(INFO) << "Trace buffers: ";
  LOG(INFO) << traceBuffers_.get();
  LOG(INFO) << traceBuffers_->cpu.size();
  traceBuffers_->cpu.push_back(std::move(cpuTrace));
}

// This has dependence on CuptiActivityInterface
CuptiActivityProfiler::CuptiActivityProfiler(
    const std::string& name,
#ifdef HAS_ROCTRACER
    RoctracerActivityApi& cupti,
#else
    CuptiActivityApi& cupti,
#endif
    bool cpuOnly)
    : name_(name),
      cupti_(cupti),
      cpuOnly_{cpuOnly} {
};

void CuptiActivityProfilerSession::log(ActivityLogger& logger) {
  LOG(INFO) << "Processing " << traceBuffers_->cpu.size()
      << " CPU buffers";
  VLOG(0) << "Profile time range: " << parent_->startTime() << " - "
          << parent_->endTime();
  for (auto& cpu_trace : traceBuffers_->cpu) {
    string trace_name = cpu_trace->span.name;
    VLOG(0) << "Processing CPU buffer for " << trace_name << " ("
            << cpu_trace->span.iteration << ") - "
            << cpu_trace->activities.size() << " records";
    VLOG(0) << "Span time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    processCpuTrace(*cpu_trace, logger);
    LOGGER_OBSERVER_ADD_EVENT_COUNT(cpu_trace->activities.size());
  }

#ifdef HAS_CUPTI
  if (!cpuOnly_) {
    VLOG(0) << "Retrieving GPU activity buffers";
    traceBuffers_->gpu = cupti_.activityBuffers();
    if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead, cupti_.flushOverhead);
    }
    if (traceBuffers_->gpu) {
      const auto count_and_size = cupti_.processActivities(
          *traceBuffers_->gpu,
          std::bind(&CuptiActivityProfilerSession::handleCuptiActivity, this, std::placeholders::_1, &logger));
      LOG(INFO) << "Processed " << count_and_size.first
                << " GPU records (" << count_and_size.second << " bytes)";
    }
  }
#endif // HAS_CUPTI
#ifdef HAS_ROCTRACER
  if (!cpuOnly_) {
    VLOG(0) << "Retrieving GPU activity buffers";
    const int count = cupti_.processActivities(
        logger,
        std::bind(&CuptiActivityProfiler::cpuActivity, this, std::placeholders::_1));
    LOG(INFO) << "Processed " << count << " GPU records";
    LOGGER_OBSERVER_ADD_EVENT_COUNT(count);
  }
#endif // HAS_ROCTRACER

  finalizeTrace(*config_, logger);
  status(TraceStatus::READY);
}

CuptiActivityProfilerSession::CpuGpuSpanPair& CuptiActivityProfilerSession::recordTraceSpan(
    TraceSpan& span, int gpuOpCount) {
  TraceSpan gpu_span(gpuOpCount, span.iteration, span.name, "GPU: ");
  auto& iterations = traceSpans_[span.name];
  iterations.push_back({span, gpu_span});
  return iterations.back();
}

void CuptiActivityProfilerSession::processCpuTrace(
    libkineto::CpuTraceBuffer& cpuTrace,
    ActivityLogger& logger) {
  if (cpuTrace.activities.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }

  CpuGpuSpanPair& span_pair =
      recordTraceSpan(cpuTrace.span, cpuTrace.gpuOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act.correlationId() << ": OP " << act.activityName;
    if (config_->selectedActivityTypes().count(act.type())) {
      VLOG(2) << act.correlationId() << " logged";
      act.log(logger);
    }
    clientActivityTraceMap_[act->correlationId()] = &span_pair;
    activityMap_[act->correlationId()] = act.get();

    recordThreadInfo(act->resourceId(), act->getThreadId(), act->deviceId());
  }
  logger.handleTraceSpan(cpu_span);
}

#ifdef HAS_CUPTI
inline void CuptiActivityProfilerSession::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
  if (correlation->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
    cpuCorrelationMap_[correlation->correlationId] = correlation->externalId;
  } else {
    LOG(INFO) << "handleCorrelationActivity(" << correlation->correlationId << ")";
    userCorrelationMap_[correlation->correlationId] = correlation->externalId;
  } else {
    LOG(WARNING)
        << "Invalid CUpti_ActivityExternalCorrelation sent to handleCuptiActivity";
  }
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
  res.endTime = gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void CuptiActivityProfilerSession::GpuUserEventMap::insertOrExtendEvent(
    const ITraceActivity& userActivity,
    const ITraceActivity& gpuActivity) {
  LOG(INFO) << "insertOrExtendEvent " << userActivity.correlationId() << ": " << userActivity.name() << " -> " << gpuActivity.name();
  StreamKey key(gpuActivity.deviceId(), gpuActivity.resourceId());
  LOG(INFO) << "key: " << gpuActivity.deviceId() << ", " <<  gpuActivity.resourceId();
  CorrelationSpanMap& correlationSpanMap = streamSpanMap_[key];
  auto it = correlationSpanMap.find(userActivity.correlationId());
  LOG(INFO) << "Map: " << (void*) &correlationSpanMap;
  if (it == correlationSpanMap.end()) {
    LOG(INFO) << "Insert!";
    auto it_success = correlationSpanMap.insert({
        userActivity.correlationId(), createUserGpuSpan(userActivity, gpuActivity)
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

void CuptiActivityProfilerSession::GpuUserEventMap::logEvents(ActivityLogger *logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

#ifdef HAS_CUPTI
inline bool CuptiActivityProfilerSession::outOfRange(const ITraceActivity& act) {
  bool out_of_range = act.timestamp() < parent_->startTime() ||
      (act.timestamp() + act.duration()) > parent_->endTime();
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
        << " (" << act.timestamp() << " < " << parent_->startTime() << " or "
        << (act.timestamp() + act.duration()) << " > " << parent_->endTime();
  }
  return out_of_range;
}

void CuptiActivityProfilerSession::handleRuntimeActivity(
    const CUpti_ActivityAPI* activity,
    ActivityLogger* logger) {
  if (isBlockListedRuntimeCbid(activity->cbid)) {
    return;
  }
  VLOG(2) << activity->correlationId
          << ": CUPTI_ACTIVITY_KIND_RUNTIME, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  const ITraceActivity* linked = linkedActivity(
      activity->correlationId, cpuCorrelationMap_);
  int32_t tid = activity->threadId;
  if (linked) {
    LOG(INFO) << "Linked: " << linked->correlationId();
    tid = linked->resourceId();
  }
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
}

inline void CuptiActivityProfilerSession::updateGpuNetSpan(
    const ITraceActivity& gpuOp) {
  if (!gpuOp.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
  const auto& it =
      clientActivityTraceMap_.find(gpuOp.linkedActivity()->correlationId());
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
void CuptiActivityProfilerSession::checkTimestampOrder(const ITraceActivity* act1) {
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
    LOG_FIRST_N(WARNING, 10) << "GPU op timestamp (" << act2->timestamp()
                             << ") < runtime timestamp (" << act1->timestamp() << ") by "
                             << act1->timestamp() - act2->timestamp() << "us" << std::endl
                             << "Name: " << act2->name() << " Device: " << act2->deviceId()
                             << " Stream: " << act2->resourceId();
  }
}

inline void CuptiActivityProfilerSession::handleGpuActivity(
    const ITraceActivity& act,
    ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  VLOG(2) << act.correlationId() << ": " << act.name();
  recordStream(act.deviceId(), act.resourceId(), "");
  act.log(*logger);
  updateGpuNetSpan(act);
  if (config_->selectedActivityTypes().count(ActivityType::GPU_USER_ANNOTATION)) {
    LOG(INFO) << "GPU_USER_ANNOTATION: " << act.correlationId();
    const auto& it = userCorrelationMap_.find(act.correlationId());
    if (it != userCorrelationMap_.end()) {
      const auto& it2 = activityMap_.find(it->second);
      if (it2 != activityMap_.end()) {
        recordStream(act.deviceId(), act.resourceId(), "context");
        gpuUserEventMap_.insertOrExtendEvent(*it2->second, act);
      }
    }
  }
}

const ITraceActivity* CuptiActivityProfilerSession::linkedActivity(
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
inline void CuptiActivityProfilerSession::handleGpuActivity(
    const T* act, ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      act->correlationId, cpuCorrelationMap_);
  const auto& gpu_activity =
      traceBuffers_->addActivityWrapper(GpuActivity<T>(act, linked));
  handleGpuActivity(gpu_activity, logger);
}

void CuptiActivityProfilerSession::handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record));
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
    case CUPTI_ACTIVITY_KIND_OVERHEAD:
      handleOverheadActivity(
          reinterpret_cast<const CUpti_ActivityOverhead*>(record), logger);
      break;
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->kind;
      break;
  }
}
#endif // HAS_CUPTI

std::shared_ptr<IActivityProfilerSession> CuptiActivityProfiler::configure(
    const Config& config,
    ICompositeProfilerSession* parentSession) {
  LOG(INFO) << "Configure!!";
  auto session = std::make_shared<CuptiActivityProfilerSession>(
      cupti_, config, cpuOnly_, parentSession);
  if (isActive() && config.hasProfileStartTime()) {
    LOG(ERROR) << "CuptiActivityProfiler already busy, terminating";
    session->status(TraceStatus::ERROR);
    return session;
  }

  if (isActive()) {
    LOG(ERROR) << "Session pre-empted";
    stop(*session_);
    session_->status(TraceStatus::ERROR);
  }

  if (config.hasProfileStartTime()) {
    int64_t start_time = toMilliseconds(timeSinceEpoch(config.requestTimestamp()));
    int64_t now = toMilliseconds(timeSinceEpoch());
    LOG(INFO) << "start: " << start_time << ", now: " << now;

    if (start_time < now) {
      LOG(ERROR) << "Not starting tracing - start timestamp is in the past. "
                 << "Time difference (ms): " << (now - start_time);
      session->status(TraceStatus::ERROR);
      return session;
    } else if ((start_time - now) <
        duration_cast<milliseconds>(config.activitiesWarmupDuration()).count()) {
      LOG(ERROR) << "Not starting tracing - insufficient time for warmup. "
                 << "Time to warmup (ms): " << (start_time - now);
      session->status(TraceStatus::ERROR);
      return session;
    } else {
      LOG(INFO) << "Tracing starting in "
                << (start_time - now) / 1000 << " secs";
    }
  }

  if (LOG_IS_ON(INFO)) {
    config.printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }

#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (!cpuOnly_) {
    // Enabling CUPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling GPU tracing";
    cupti_.setMaxBufferSize(config.activitiesMaxGpuBufferSize());

    cupti_.clearActivities();
    time_point<system_clock> t;
    if (VLOG_IS_ON(1)) {
      t = system_clock::now();
    }
#ifdef HAS_CUPTI
    cupti_.enableCuptiActivities(config.selectedActivityTypes());
#else
    cupti_.enableActivities(config.selectedActivityTypes());
#endif
    if (VLOG_IS_ON(1)) {
      session->addOverheadSample(
          session->setupOverhead,
          duration_cast<microseconds>(system_clock::now() - t).count());
    }
    LOG(INFO) << "Doine";
  }
#endif // HAS_CUPTI || HAS_ROCTRACER

  parentSession->registerCorrelationObserver(&cupti_);
  session->status(TraceStatus::WARMUP);
  session_ = std::move(session);
  return session_;
}

void CuptiActivityProfiler::start(IActivityProfilerSession& session) {
  LOG(INFO) << "Session: " << (void*) &session;
  if (&session != session_.get()) {
    LOG(ERROR) << "invalid session";
    session.status(TraceStatus::ERROR);
    return;
  }
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (cupti_.error()) {
    session.status(TraceStatus::ERROR);
    session_ = nullptr;
    return;
  }
#endif
  session_->start();
}

void CuptiActivityProfiler::stop(IActivityProfilerSession& session) {
  LOG(INFO) << "Session: " << (void*) &session;
  if (&session != session_.get()) {
    LOG(ERROR) << "Invalid session";
    session.status(TraceStatus::ERROR);
  } else if (session.status() != TraceStatus::RECORDING) {
    LOG(ERROR) << "Invalid status: " << (int) session.status();
    session.status(TraceStatus::ERROR);
  } else {
    LOG(INFO) << "STOP";
    session_->stop();
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
    if (!cpuOnly_) {
      time_point<system_clock> t;
      if (VLOG_IS_ON(1)) {
        t = system_clock::now();
      }
      cupti_.disableCuptiActivities();
      if (VLOG_IS_ON(1)) {
        session_->addOverheadSample(
            session_->setupOverhead,
            duration_cast<microseconds>(system_clock::now() - t).count());
      }
    }
#endif // HAS_CUPTI || HAS_ROCTRACER
  }
  LOG(INFO) << "Stopped";
}

void CuptiActivityProfilerSession::finalizeTrace(const Config& config, ActivityLogger& logger) {
  LOG(INFO) << "Recorded nets:";
  {
    for (const auto& it : iterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    iterationCountMap_.clear();
  }

  if (!cpuOnly_ && parent_) {
    // GPU events use device id as pid (0-7).
    constexpr int kMaxGpuCount = 8;
    const DeviceInfo pid_info = parent_->deviceInfo(processId());
    for (int gpu = 0; gpu < kMaxGpuCount; gpu++) {
      parent_->recordDeviceInfo(gpu, pid_info.name, fmt::format("GPU {}", gpu));
    }
  }

  for (const auto& iterations : traceSpans_) {
    for (const auto& span_pair : iterations.second) {
      const TraceSpan& gpu_span = span_pair.second;
      if (gpu_span.opCount > 0) {
        logger.handleTraceSpan(gpu_span);
      }
    }
  }

  // Overhead info
  overheadInfo_.push_back(ActivityLogger::OverheadInfo("CUPTI Overhead"));
  for(const auto& info : overheadInfo_) {
    logger.handleOverheadInfo(info, captureWindowStartTime_);
  }

  gpuUserEventMap_.logEvents(&logger);
}

} // namespace KINETO_NAMESPACE
