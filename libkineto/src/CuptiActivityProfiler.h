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

#include "ThreadUtil.h"
#include "TraceSpan.h"
#include "libkineto.h"
#include "output_base.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"

namespace KINETO_NAMESPACE {

class Config;
class CuptiActivityApi;
class RoctracerActivityApi;

class CuptiActivityProfiler {
 public:
  CuptiActivityProfiler(CuptiActivityApi& cupti, bool cpuOnly);
  CuptiActivityProfiler(RoctracerActivityApi& rai, bool cpuOnly);
  CuptiActivityProfiler(const CuptiActivityProfiler&) = delete;
  CuptiActivityProfiler& operator=(const CuptiActivityProfiler&) = delete;

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

  // Synchronous control API
  void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    startTraceInternal(now);
  }

  void stopTrace(const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    stopTraceInternal(now);
  }

  // Process CPU and GPU traces
  void processTrace(ActivityLogger& logger) {
    std::lock_guard<std::mutex> guard(mutex_);
    processTraceInternal(logger);
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    resetInternal();
  }

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
    return applyNetFilterInternal(name);
  }

  bool applyNetFilterInternal(const std::string& name);

  Config& config() {
    return *config_;
  }

  inline void recordThreadInfo() {
    int32_t sysTid = systemThreadId();
    // Note we're using the lower 32 bits of the (opaque) pthread id
    // as key, because that's what CUPTI records.
    int32_t tid = threadId();
    int32_t pid = processId();
    std::lock_guard<std::mutex> guard(mutex_);
    if (resourceInfo_.find({pid, tid}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(pid, tid),
          ActivityLogger::ResourceInfo(
              pid,
              sysTid,
              fmt::format("thread {} ({})", sysTid, getThreadName())));
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> guard(mutex_);
    metadata_[key] = value;
  }

  void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) {
    std::lock_guard<std::mutex> guard(mutex_);
    profilers_.push_back(std::move(profiler));
  }

 protected:

  using CpuGpuSpanPair = std::pair<TraceSpan, TraceSpan>;
  static const CpuGpuSpanPair& defaultTraceSpan();

 private:

  class ExternalEventMap {
   public:

    // The correlation id of the GPU activity
    const libkineto::GenericTraceActivity& correlatedActivity(
        uint32_t correlation_id);
    void insertEvent(const libkineto::GenericTraceActivity* op);

    void addCorrelation(uint64_t external_id, uint32_t cuda_id);

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
    std::unordered_map<int64_t, const libkineto::GenericTraceActivity*>
        events_;

    // Cuda correlation id -> external correlation id for default events
    // CUPTI provides a mechanism for correlating Cuda events to arbitrary
    // external events, e.g.operator events from Caffe2.
    // It also marks GPU activities with the Cuda event correlation ID.
    // So by connecting the two, we get the complete picture.
    std::unordered_map<
        uint32_t, // Cuda correlation ID
        uint64_t> // External correlation ID
        correlationMap_;
  };

  // Map of gpu activities to user defined events
  class GpuUserEventMap {
   public:
    // Insert a user defined event which maps to the gpu trace activity.
    // If the user defined event mapping already exists this will update the
    // gpu side span to include the span of gpuTraceActivity.
    void insertOrExtendEvent(const TraceActivity& cpuTraceActivity,
      const TraceActivity& gpuTraceActivity);
    // Log out the events to the logger
    void logEvents(ActivityLogger *logger);

    void clear() {
      streamSpanMap_.clear();
    }

   private:
    // device id and stream name
    using StreamKey = std::pair<int64_t, int64_t>;

    // map of correlation id to TraceSpan
    using CorrelationSpanMap =
        std::unordered_map<int64_t, GenericTraceActivity>;
    std::map<StreamKey, CorrelationSpanMap> streamSpanMap_;
  };

  GpuUserEventMap gpuUserEventMap_;

  // data structure to collect cuptiActivityFlushAll() latency overhead
  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

  void startTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void stopTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void processTraceInternal(ActivityLogger& logger);

  void resetInternal();

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  void configureChildProfilers();

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

  // Create resource names for streams
  inline void recordStream(int device, int id) {
    if (resourceInfo_.find({device, id}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(device, id),
          ActivityLogger::ResourceInfo(
              device, id, fmt::format("stream {}", id)));
    }
  }

  // Record client trace span for subsequent lookups from activities
  // Also creates a corresponding GPU-side span.
  CpuGpuSpanPair& recordTraceSpan(TraceSpan& span, int gpuOpCount);

  // Returns true if net name is to be tracked for a specified number of
  // iterations.
  bool iterationTargetMatch(libkineto::CpuTraceBuffer& trace);

  // net name to id
  int netId(const std::string& netName);

#ifdef HAS_CUPTI
  // Process generic CUPTI activity
  void handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger);

  // Process specific GPU activity types
  void updateGpuNetSpan(const TraceActivity& gpuOp);
  bool outOfRange(const TraceActivity& act);
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity, ActivityLogger* logger);
  void handleGpuActivity(const TraceActivity& act,
      ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);
#endif // HAS_CUPTI

  // Is logging disabled for this event?
  // Logging can be disabled due to operator count, net name filter etc.
  inline bool loggingDisabled(const libkineto::TraceActivity& act) const {
    const auto& it = clientActivityTraceMap_.find(act.correlationId());
    return it != clientActivityTraceMap_.end() &&
        disabledTraceSpans_.find(it->second->first.name) !=
        disabledTraceSpans_.end();
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
#ifdef HAS_ROCTRACER
  RoctracerActivityApi& cupti_;		// Design failure here
#else
  CuptiActivityApi& cupti_;
#endif

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
  using ActivityTraceMap = std::unordered_map<int64_t, CpuGpuSpanPair*>;
  ActivityTraceMap clientActivityTraceMap_;

  // Cache thread names and system thread ids for pthread ids,
  // and stream ids for GPU streams
  std::map<
      std::pair<int64_t, int64_t>,
      ActivityLogger::ResourceInfo> resourceInfo_;

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

  // Trace metadata
  std::unordered_map<std::string, std::string> metadata_;

  // child activity profilers
  std::vector<std::unique_ptr<IActivityProfiler>> profilers_;

  // a vector of active profiler plugin sessions
  std::vector<std::unique_ptr<IActivityProfilerSession>> sessions_;
};

} // namespace KINETO_NAMESPACE
