/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cupti.h>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ThreadName.h"
#include "TraceSpan.h"
#include "libkineto.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

class Config;
class CuptiActivityInterface;

class ActivityProfiler {
 public:
  ActivityProfiler(CuptiActivityInterface& cupti, bool cpuOnly);
  ActivityProfiler(const ActivityProfiler&) = delete;
  ActivityProfiler& operator=(const ActivityProfiler&) = delete;

  bool isActive() const {
    return currentRunloopState_ != RunloopState::WaitForRequest;
  }

  // Invoke at a regular interval to perform profiling activities.
  // When not active, an interval of 1-5 seconds is probably fine,
  // depending on required warm-up time and delayed start time.
  // When active, it's a good idea to invoke more frequently to stay below
  // memory usage limit (ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB) during warmup.
  const std::chrono::time_point<std::chrono::system_clock> performRunLoopStep(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime);

  // Used for async requests
  void setLogger(ActivityLogger* logger) {
    logger_ = logger;
  }

  // Explicit control
  // FIXME: Refactor out profiler loop
  void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    startTraceUnlocked(now);
  }

  void stopTrace(const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    stopTraceUnlocked(now);
  }

  // Process CPU and GPU traces
  void processTrace(ActivityLogger& logger);

  void cancelTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void resetTrace();

  // Set up profiler as specified in config.
  void configure(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  // Registered with client API to pass CPU trace events over
  void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace);

  // Registered with external API so that CPU-side tracer can filter which nets
  // to trace
  bool applyNetFilter(const std::string& name) {
    std::lock_guard<std::mutex> guard(mutex_);
    return applyNetFilterUnlocked(name);
  }

  bool applyNetFilterUnlocked(const std::string& name);

  Config& config() {
    return *config_;
  }

 private:
  class ExternalEventMap {
   public:
    const libkineto::ClientTraceActivity& operator[](uint32_t id);
    void insertEvent(const libkineto::ClientTraceActivity* op);

    void addCorrelation(uint64_t external_id, uint32_t cuda_id) {
      correlationMap_[cuda_id] = external_id;
    }

    void clear() {
      events_.clear();
      correlationMap_.clear();
    }

   private:
    // Map extern correlation ID to Operator info.
    // This is a map of regular pointers which is generally a bad idea,
    // but this class also fully owns the objects it is pointing to so
    // it's not so bad. This is done for performance reasons and is an
    // implementation detail of this class that might change.
    std::unordered_map<uint64_t, const libkineto::ClientTraceActivity*>
        events_;

    // Cuda correlation id -> external correlation id
    // CUPTI provides a mechanism for correlating Cuda events to arbitrary
    // external events, e.g.operator events from Caffe2.
    // It also marks GPU activities with the Cuda event correlation ID.
    // So by connecting the two, we get the complete picture.
    std::unordered_map<
        uint32_t, // Cuda correlation ID
        uint64_t> // External correlation ID
        correlationMap_;
  };

  // data structure to collect cuptiActivityFlushAll() latency overhead
  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

  void startTraceUnlocked(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void stopTraceUnlocked(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  // Process a single CPU trace
  void processCpuTrace(
      libkineto::CpuTraceBuffer& cpuTrace,
      ActivityLogger& logger,
      bool logNet);

  bool inline passesGpuOpCountThreshold(
      const libkineto::CpuTraceBuffer& cpuTrace) {
    return cpuOnly_ || cpuTrace.gpuOpCount < 0 ||
        cpuTrace.gpuOpCount >= netGpuOpCountThreshold_;
  }

  // Record client trace span for subsequent lookups from activities
  // Also creates a corresponding GPU-side span.
  using CpuGpuSpanPair = std::pair<TraceSpan, TraceSpan>;
  CpuGpuSpanPair& recordTraceSpan(TraceSpan& span, int gpuOpCount);

  // Returns true if net name is to be tracked for a specified number of
  // iterations.
  bool iterationTargetMatch(
      const libkineto::CpuTraceBuffer& trace);

  // net name to id
  int netId(const std::string& netName);

  // Process generic CUPTI activity
  void handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger);

  // Process specific GPU activity types
  void updateGpuNetSpan(const TraceActivity& gpuOp);
  bool outOfRange(const TraceActivity& act);
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity, ActivityLogger* logger);
  void handleGpuActivity(const TraceActivity& act, ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);

  // Is logging disabled for this event?
  // Logging can be disabled due to operator count, net name filter etc.
  inline bool loggingDisabled(const libkineto::TraceActivity& act) {
    const auto& it = clientActivityTraceMap_.find(act.correlationId());
    return it != clientActivityTraceMap_.end() &&
        disabledTraceSpans_.find(it->second->first.name) !=
        disabledTraceSpans_.end();
  }

  inline void recordThreadName(pthread_t pthreadId) {
    if (threadNames_.find(pthreadId) == threadNames_.end()) {
      threadNames_[pthreadId] = getThreadName(pthreadId);
    }
  }

  void resetTraceData();

  void addOverheadSample(profilerOverhead& counter, int64_t overhead) {
    counter.overhead += overhead;
    counter.cntr++;
  }
  int64_t getOverhead(const profilerOverhead& counter) {
    if (counter.cntr == 0) {
      return 0;
    }
    return counter.overhead / counter.cntr;
  }

  // On-demand request configuration
  std::unique_ptr<Config> config_;

  // Logger used during trace processing
  ActivityLogger* logger_;

  // Calls to CUPTI is encapsulated behind this interface
  CuptiActivityInterface& cupti_;

  enum class RunloopState {
    WaitForRequest,
    Warmup,
    CollectTrace,
    ProcessTrace
  };

  // Start and end time used for triggering and stopping profiling
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  std::chrono::time_point<std::chrono::system_clock> profileEndTime_;

  ExternalEventMap externalEvents_;

  // All recorded trace spans, both CPU and GPU
  // Trace Id -> list of iterations.
  // Using map of lists for the iterator semantics, since we are recording
  // pointers to the elements in this structure.
  std::map<std::string, std::list<CpuGpuSpanPair>> traceSpans_;

  // Maintain a map of client trace activity to trace span.
  // Maps correlation id -> TraceSpan* held by traceSpans_.
  std::unordered_map<int64_t, CpuGpuSpanPair*> clientActivityTraceMap_;

  // Cache thread names for pthread ids
  std::unordered_map<uint64_t, std::string> threadNames_;

  // Which trace spans are disabled. Together with the operator -> net id map
  // this allows us to determine whether a GPU or CUDA API event should
  // be included in the trace.
  // If a CUDA event cannot be mapped to a net it will always be included.
  std::unordered_set<std::string> disabledTraceSpans_;

  // the overhead to flush the activity buffer
  profilerOverhead flushOverhead_;
  // the overhead to enable/disable activity tracking
  profilerOverhead setupOverhead_;

  bool cpuOnly_{false};

  // ***************************************************************************
  // Below state is shared with external threads.
  // These need to either be atomic, accessed under lock or only used
  // by external threads in separate runloop phases from the profiler thread.
  // ***************************************************************************

  // Mutex to protect non-atomic access to below state
  std::mutex mutex_;

  // Runloop phase
  std::atomic<RunloopState> currentRunloopState_{RunloopState::WaitForRequest};

  // Keep track of the start time of the first net in the current trace.
  // This is only relevant to Caffe2 as PyTorch does not have nets.
  // All CUDA events before this time will be removed
  // Can be written by external threads during collection.
  int64_t captureWindowStartTime_{0};
  // Similarly, all CUDA API events after the last net event will be removed
  int64_t captureWindowEndTime_{0};

  // net name -> iteration count
  std::map<std::string, int> netIterationCountMap_;
  // Sub-strings used to filter nets by name
  std::vector<std::string> netNameFilter_;
  // Filter by GPU op count
  int netGpuOpCountThreshold_{0};
  // Net used to track iterations
  std::string netIterationsTarget_;
  // Number of iterations to track
  int netIterationsTargetCount_{0};

  // Flag used to stop tracing from external api callback.
  // Needs to be atomic since it's set from a different thread.
  std::atomic_bool stopCollection_{false};

  // Buffers where trace data is stored
  std::unique_ptr<ActivityBuffers> traceBuffers_;

};

} // namespace KINETO_NAMESPACE
