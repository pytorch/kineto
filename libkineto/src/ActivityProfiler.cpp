/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfiler.h"

#include <fmt/format.h>
#include <libgen.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

#include <cupti.h>

#include "Config.h"
#include "CuptiActivityInterface.h"
#include "output_base.h"

#include "Logger.h"

using namespace std::chrono;
using namespace libkineto;
using std::string;

namespace KINETO_NAMESPACE {

bool ActivityProfiler::iterationTargetMatch(
    const external_api::CpuTraceBuffer& trace) {
  bool match = (trace.netName == netIterationsTarget_);
  if (!match && external_api::enableForNet(trace.netName) &&
      passesGpuOpCountThreshold(trace)) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (netIterationsTarget_.empty()) {
      match = true;
      LOG(INFO) << "Target net for iterations not specified "
                << "- picking first encountered that passes net filter";
    } else if (trace.netName.find(netIterationsTarget_) != trace.netName.npos) {
      // Only track the first one that matches
      match = true;
    }
    if (match) {
      netIterationsTarget_ = trace.netName;
      LOG(INFO) << "Tracking net " << trace.netName << " for "
                << netIterationsTargetCount_ << " iterations";
    }
  }
  return match;
}

void ActivityProfiler::transferCpuTrace(
    std::unique_ptr<external_api::CpuTraceBuffer> cpuTrace) {
  // FIXME: It's theoretically possible to receive a buffer from a
  // previous trace request. Probably should add a serial number.
  if (currentRunloopState_ != RunloopState::CollectTrace &&
      currentRunloopState_ != RunloopState::ProcessTrace) {
    VLOG(0) << "Trace collection not in progress - discarding trace of net "
            << cpuTrace->netName;
    return;
  }

  int iteration;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // Count iterations per net and stop profiling if the target net
    // has reached the iteration target (if no target net has been set,
    // one is picked at random)
    iteration = netIterationCountMap_[cpuTrace->netName]++;
  }

  VLOG(0) << "Received iteration " << iteration << " of net "
          << cpuTrace->netName << " (" << cpuTrace->ops.size() << " ops / "
          << cpuTrace->gpuOpCount << " gpu ops)";
  if (currentRunloopState_ == RunloopState::CollectTrace &&
      iterationTargetMatch(*cpuTrace)) {
    if (iteration == 0) {
      VLOG(0) << "Setting profile start time from net to "
              << cpuTrace->startTime;
      captureWindowStartTime_ = cpuTrace->startTime;
    } else if (1 + iteration >= netIterationsTargetCount_) {
      VLOG(0) << "Completed target iteration count for net "
              << cpuTrace->netName;
      external_api::setProfileRequestActive(false);
      // Tell the runloop to stop collection
      stopCollection_ = true;
      captureWindowEndTime_ = cpuTrace->endTime;
    }
  }

  cpuTraceQueue_.emplace(iteration, std::move(cpuTrace));
}

bool ActivityProfiler::applyNetFilter(const std::string& name) {
  std::lock_guard<std::mutex> guard(mutex_);
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

void ActivityProfiler::processTraces() {
  // A CPU trace needs to be processed before the corresponding GPU trace.
  // When entering this function it is assumed that if GPU buffers exist,
  // then the corresponding CPU buffers also exist. It's OK if additional
  // CPU buffers appear while processing is in progress. It's also guaranteed
  // New GPU buffers should only appear via a cuptiActivityFlushAll() call
  // from the same thread.
  mutex_.lock();
  LOG(INFO) << "Processing " << cpuTraceQueue_.size() << " CPU buffers";
  VLOG(0) << "Profile time range: " << captureWindowStartTime_ << " - "
          << captureWindowEndTime_;
  while (!cpuTraceQueue_.empty()) {
    auto& pair = cpuTraceQueue_.front();
    int instance = pair.first;
    auto cpu_trace = std::move(pair.second);
    cpuTraceQueue_.pop();
    mutex_.unlock();
    VLOG(0) << "Processing CPU buffer for " << cpu_trace->netName << " ("
            << instance << ") - " << cpu_trace->ops.size() << " records";
    bool log_net = external_api::enableForNet(cpu_trace->netName) &&
        passesGpuOpCountThreshold(*cpu_trace) &&
        cpu_trace->startTime < captureWindowEndTime_ &&
        cpu_trace->endTime > captureWindowStartTime_;
    VLOG(0) << "Net time range: " << cpu_trace->startTime << " - "
            << cpu_trace->endTime;
    VLOG(0) << "Log net: " << (log_net ? "Yes" : "No");
    processCpuTrace(instance, std::move(cpu_trace), log_net);
    mutex_.lock();
  }
  mutex_.unlock();

  if (!cpuOnly_) {
    const auto count_and_size = cupti_.processActivities(std::bind(&ActivityProfiler::handleCuptiActivity, this, std::placeholders::_1));
    LOG(INFO) << "Processed " << count_and_size.first << " GPU records (" << count_and_size.second << " bytes)";
    if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead_, cupti_.flushOverhead);
    }
  }
}

int ActivityProfiler::netId(const std::string& netName) {
  static int cur_net_id = 0;
  const auto& it = netIdMap_.find(netName);
  if (it != netIdMap_.end()) {
    return it->second;
  } else {
    netNames_.push_back(netName);
    return netIdMap_[netName] = cur_net_id++;
  }
}

void ActivityProfiler::processCpuTrace(
    int instance,
    std::unique_ptr<external_api::CpuTraceBuffer> cpuTrace,
    bool logNet) {
  if (cpuTrace->ops.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }
  int net_id = netId(cpuTrace->netName);
  for (auto const& op_stat : cpuTrace->ops) {
    VLOG(2) << op_stat.correlationId << ": OP " << op_stat.opType
            << " tid: " << op_stat.threadId;
    if (logNet) {
      logger_->handleCpuActivity(cpuTrace->netName, instance, op_stat);
      recordThreadName(op_stat.threadId);
    }
    // Stash event so we can look it up later when processing GPU trace
    externalEvents_.insertEvent(&op_stat);
    opNetMap_[op_stat.correlationId] = {net_id, instance};
  }
  if (logNet) {
    logger_->handleNetCPUSpan(
        net_id,
        cpuTrace->netName,
        instance,
        cpuTrace->ops.size(),
        cpuTrace->gpuOpCount,
        cpuTrace->startTime,
        cpuTrace->endTime);
    if (cpuTrace->netName == netIterationsTarget_) {
      size_t tid = cpuTrace->ops[0].threadId;
      logger_->handleIterationStart(
          cpuTrace->netName, cpuTrace->startTime, tid);
    }
  } else {
    externalDisabledNets_.insert(net_id);
  }
  externalEvents_.addTraceData(std::move(cpuTrace));
}

inline void ActivityProfiler::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
  externalEvents_.addCorrelation(
      correlation->externalId, correlation->correlationId);
  VLOG(2) << correlation->correlationId
          << ": CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION";
}

void ActivityProfiler::ExternalEventMap::insertEvent(
    const libkineto::external_api::OpDetails* op) {
  if (events_[op->correlationId] != nullptr) {
    LOG_EVERY_N(WARNING, 100)
        << "Events processed out of order - link will be missing";
  }
  events_[op->correlationId] = op;
}

static uint64_t usecs(uint64_t nsecs) {
  return nsecs / 1000;
}

inline bool ActivityProfiler::outOfRange(
    uint64_t startNsecs,
    uint64_t endNsecs) {
  return usecs(endNsecs) < captureWindowStartTime_ ||
      usecs(startNsecs) > captureWindowEndTime_;
}

inline void ActivityProfiler::handleRuntimeActivity(
    const CUpti_ActivityAPI* activity) {
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
  const external_api::OpDetails& ext = externalEvents_[activity->correlationId];
  if (ext.correlationId == 0 && outOfRange(activity->start, activity->end)) {
    return;
  }
  if (!loggingDisabled(ext)) {
    logger_->handleRuntimeActivity(activity, ext);
    // FIXME: This is pretty hacky and it's likely that we miss some links.
    // May need to maintain a map instead.
    if (activity->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
        (activity->cbid >= CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
         activity->cbid <= CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020) ||
        activity->cbid ==
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
        activity->cbid ==
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000) {
      logger_->handleLinkStart(activity);
    }
  }
}

inline void ActivityProfiler::updateGpuNetSpan(
    uint64_t startUsecs,
    uint64_t endUsecs,
    const external_api::OpDetails& ext) {
  const auto& it = opNetMap_.find(ext.correlationId);
  if (it == opNetMap_.end()) {
    // No correlation id mapping?
    return;
  }
  const NetId& id = it->second;
  auto& instances = gpuNetSpanMap_[id.net];
  while (id.instance >= instances.size()) {
    instances.push_back({});
  }
  auto& span = gpuNetSpanMap_[id.net][id.instance];
  if (startUsecs < span.first || span.first == 0) {
    span.first = startUsecs;
  }
  if (endUsecs > span.second) {
    span.second = endUsecs;
  }
}

// I've observed occasional broken timestamps attached to GPU events...
static const string gpuOpName(const CUpti_ActivityKernel4& kernel) {
  return string("Kernel: ") + kernel.name;
}
static const string gpuOpName(const CUpti_ActivityMemcpy& /* unused */) {
  return "Memcpy";
}
static const string gpuOpName(const CUpti_ActivityMemcpy2& /* unused */) {
  return "Memcpy2";
}
static const string gpuOpName(const CUpti_ActivityMemset& /* unused */) {
  return "Memset";
}
template <class T>
static bool timestampsInCorrectOrder(
    const external_api::OpDetails& ext,
    const T& gpuOp) {
  if (ext.startTime > usecs(gpuOp.start)) {
    LOG(WARNING) << "GPU op timestamp (" << usecs(gpuOp.start)
                 << ") < runtime timestamp (" << ext.startTime << ")";
    LOG(WARNING) << "Name: " << gpuOpName(gpuOp)
                 << " Device: " << gpuOp.deviceId
                 << " Stream: " << gpuOp.streamId;
    return false;
  }
  return true;
}

// FIXME: Unify with below
inline void ActivityProfiler::handleGpuActivity(
    const CUpti_ActivityKernel4* kernel) {
  const external_api::OpDetails& ext = externalEvents_[kernel->correlationId];
  if (ext.startTime == 0 && outOfRange(kernel->start, kernel->end)) {
    return;
  }
  if (!timestampsInCorrectOrder(ext, *kernel)) {
    return;
  }

  VLOG(2) << ext.correlationId << "," << kernel->correlationId
          << ": CONC KERNEL - ext. corr=";
  if (!loggingDisabled(ext)) {
    logger_->handleGpuActivity(kernel, ext, cupti_.smCount());
    // Create flow from external event
    logger_->handleLinkEnd(
        kernel->correlationId,
        kernel->deviceId,
        kernel->streamId,
        usecs(kernel->start));
    updateGpuNetSpan(usecs(kernel->start), usecs(kernel->end), ext);
  }
}

template <class T>
inline void ActivityProfiler::handleGpuActivity(
    const T* act,
    const char* name) {
  const external_api::OpDetails& ext = externalEvents_[act->correlationId];
  if (ext.startTime == 0 && outOfRange(act->start, act->end)) {
    return;
  }
  if (!timestampsInCorrectOrder(ext, *act)) {
    return;
  }

  // FIXME: use typeid
  VLOG(2) << ext.correlationId << "," << act->correlationId << ": " << name;
  if (!loggingDisabled(ext)) {
    logger_->handleGpuActivity(act, ext);
    // Create flow from external event
    logger_->handleLinkEnd(
        act->correlationId, act->deviceId, act->streamId, usecs(act->start));
    updateGpuNetSpan(usecs(act->start), usecs(act->end), ext);
  }
}

void ActivityProfiler::handleCuptiActivity(const CUpti_Activity* record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(
              record));
      break;
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const CUpti_ActivityAPI*>(record));
      break;
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityKernel4*>(record));
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpy*>(record), "MEMCPY");
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpy2*>(record),
          "MEMCPY2");
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemset*>(record), "MEMSET");
      break;
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->kind;
      break;
  }
}

void ActivityProfiler::configure(
    Config& config,
    std::unique_ptr<ActivityLogger> logger,
    const time_point<system_clock>& now) {
  config_ = config.clone();
  logger_ = std::move(logger);
  logger_->configure(*config_);

  if (config_->activitiesOnDemandDuration().count() == 0) {
    // Use default if not specified
    config_->setActivitiesOnDemandDuration(
        config_->activitiesOnDemandDurationDefault());
  }

  config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  if (!cpuOnly_ && !external_api::isSupported()) {
    LOG(INFO) << "GPU-only tracing for "
              << config_->activitiesOnDemandDuration().count() << "ms";
  } else {
    std::lock_guard<std::mutex> guard(mutex_);
    netNameFilter_ = config_->activitiesOnDemandExternalFilter();
    netGpuOpCountThreshold_ =
        config_->activitiesOnDemandExternalGpuOpCountThreshold();
    netIterationsTarget_ = config_->activitiesOnDemandExternalTarget();
    external_api::setNetSizeThreshold(
        config_->activitiesOnDemandExternalNetSizeThreshold());
    netIterationsTargetCount_ = config_->activitiesOnDemandExternalIterations();

    // Clear any late arrivers from previous request
    while (!cpuTraceQueue_.empty()) {
      cpuTraceQueue_.pop();
    }
  }

  if (!cpuOnly_) {
    // Enabling CUPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling GPU tracing";
    cupti_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());

    time_point<high_resolution_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = high_resolution_clock::now();
    }
    cupti_.enableCuptiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = high_resolution_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }

  std::lock_guard<std::mutex> guard(mutex_);
  profileStartTime_ = (config_->requestTimestamp() + config_->maxRequestAge()) +
      config_->activitiesWarmupDuration();
  if (profileStartTime_ < now) {
    profileStartTime_ = now + config_->activitiesWarmupDuration();
  }
  LOG(INFO) << "Tracing starting in "
            << duration_cast<seconds>(profileStartTime_ - now).count() << "s";

  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  currentRunloopState_ = RunloopState::Warmup;
}

void ActivityProfiler::endTrace() {
  external_api::setProfileRequestActive(false);
  if (!cpuOnly_) {
    time_point<high_resolution_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = high_resolution_clock::now();
    }
    cupti_.disableCuptiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = high_resolution_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
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
      // Flushing can take a while so avoid doing it close to the start time
      if (!cpuOnly_ && nextWakeupTime < profileStartTime_) {
        cupti_.clearActivities();
      }

      if (cupti_.stopCollection) {
        // Go to process trace to clear any outstanding buffers etc
        endTrace();
        currentRunloopState_ = RunloopState::ProcessTrace;
      }

      if (now >= profileStartTime_) {
        if (now > profileStartTime_ + milliseconds(10)) {
          LOG(WARNING)
              << "Tracing started "
              << duration_cast<milliseconds>(now - profileStartTime_).count()
              << "ms late!";
        } else {
          LOG(INFO) << "Tracing started";
        }
        captureWindowStartTime_ = external_api::timeSinceEpoch(now);
        external_api::setProfileRequestActive(true);
        currentRunloopState_ = RunloopState::CollectTrace;
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
        profileEndTime_ = time_point<high_resolution_clock>(
                              microseconds(captureWindowStartTime_)) +
            config_->activitiesOnDemandDuration();
      }

      if (now >= profileEndTime_ || stopCollection_.exchange(false) ||
          cupti_.stopCollection) {
        // Update runloop state first to prevent further updates to shared state
        currentRunloopState_ = RunloopState::ProcessTrace;
        LOG(INFO) << "Tracing complete";
        if (captureWindowEndTime_ == 0) {
          captureWindowEndTime_ = external_api::timeSinceEpoch(now);
        }
        endTrace();
        VLOG_IF(0, now >= profileEndTime_) << "Reached profile end time";
        VLOG(0) << "CollectTrace -> ProcessTrace";
      } else if (now < profileEndTime_ && profileEndTime_ < nextWakeupTime) {
        new_wakeup_time = profileEndTime_;
      }

      break;

    case RunloopState::ProcessTrace:
      processTraces();
      finalizeTrace(*config_);
      resetTraceData();
      currentRunloopState_ = RunloopState::WaitForRequest;
      VLOG(0) << "ProcessTrace -> WaitForRequest";
      break;
  }

  return new_wakeup_time;
}

// Extract process name from /proc/pid/cmdline. This does not have
// the 16 character limit that /proc/pid/status and /prod/pid/comm has.
const string processName(pid_t pid) {
  FILE* cmdfile = fopen(fmt::format("/proc/{}/cmdline", pid).c_str(), "r");
  if (cmdfile != nullptr) {
    char* command = nullptr;
    int scanned = fscanf(cmdfile, "%ms", &command);
    if (scanned > 0 && command) {
      string ret(basename(command));
      free(command);
      return ret;
    }
  }
  VLOG(1) << "Failed to read process name for pid " << pid;
  return "";
}

void ActivityProfiler::finalizeTrace(const Config& config) {
  LOG(INFO) << "Recorded nets:";
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& it : netIterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    netIterationCountMap_.clear();
  }

  // Process names
  string process_name = processName(getpid());
  if (!process_name.empty()) {
    pid_t pid = getpid();
    logger_->handleProcessName(
        pid, process_name, "CPU", captureWindowStartTime_);
    if (!cpuOnly_) {
      // GPU events use device id as pid (0-7).
      constexpr int kMaxGpuCount = 8;
      for (int gpu = 0; gpu < kMaxGpuCount; gpu++) {
        logger_->handleProcessName(
            gpu,
            process_name,
            fmt::format("GPU {}", gpu),
            captureWindowStartTime_);
      }
    }
  }
  // Thread names
  for (auto pair : threadNames_) {
    logger_->handleThreadName(
        (uint32_t)pair.first, pair.second, captureWindowStartTime_);
  }

  for (const auto& netspan : gpuNetSpanMap_) {
    int id = netspan.first;
    for (int i = 0; i < netspan.second.size(); i++) {
      const auto& span = netspan.second[i];
      logger_->handleNetGPUSpan(id, netNames_[id], i, span.first, span.second);
    }
  }

  logger_->finalizeTrace(config);
}

} // namespace KINETO_NAMESPACE
