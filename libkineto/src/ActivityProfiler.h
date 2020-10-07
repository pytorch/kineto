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
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cupti.h>

#include "ThreadName.h"
#include "external_api.h"

namespace KINETO_NAMESPACE {

class ActivityLogger;
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

  // Set up profiler as specified in config.
  void configure(
      Config& config,
      std::unique_ptr<ActivityLogger> logger,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  // Registered with external API to pass CPU trace events over
  void transferCpuTrace(
      std::unique_ptr<libkineto::external_api::CpuTraceBuffer> cpuTrace);

  // Registered with external API so that CPU-side tracer can filter which nets
  // to trace
  bool applyNetFilter(const std::string& name);

 private:
  class ExternalEventMap {
   public:
    const libkineto::external_api::OpDetails& operator[](uint32_t id) {
      auto* res = events_[correlationMap_[id]];
      if (res == nullptr) {
        // Entry may be missing because cpu trace hasn't been processed yet
        // Insert a dummy element so that we can check for this in insertEvent
        static const libkineto::external_api::OpDetails nullOp_{};
        events_[correlationMap_[id]] = &nullOp_;
        res = &nullOp_;
      }
      return *res;
    }

    void insertEvent(const libkineto::external_api::OpDetails* op);

    void addCorrelation(uint64_t external_id, uint32_t cuda_id) {
      correlationMap_[cuda_id] = external_id;
    }

    void addTraceData(
        std::unique_ptr<libkineto::external_api::CpuTraceBuffer> cpuTrace) {
      cpuTraces_.push_back(std::move(cpuTrace));
    }

    void clear() {
      events_.clear();
      correlationMap_.clear();
      cpuTraces_.clear();
    }

   private:
    // Map extern correlation ID to Operator info.
    // This is a map of regular pointers which is generally a bad idea,
    // but this class also fully owns the objects it is pointing to so
    // it's not so bad. This is done for performance reasons and is an
    // implementation detail of this class that might change.
    std::unordered_map<uint64_t, const libkineto::external_api::OpDetails*>
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

    // Processed traces go here - they are kept around since pointers to
    // the elements are stored in the event map and are accessed when the
    // GPU trace is processed at some later time. Once we know it's safe,
    // these are deleted.
    std::vector<std::unique_ptr<libkineto::external_api::CpuTraceBuffer>>
        cpuTraces_;
  };

  // data structure to collect cuptiActivityFlushAll() latency overhead
  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

  // Stop tracing
  void endTrace();

  // Compress and upload the trace to Manifold if configured
  void finalizeTrace(const Config& config);

  // Process the CPU and GPU traces in the queue, start merging them if we can
  void processTraces();

  // Process a single CPU trace
  void processCpuTrace(
      int instance,
      std::unique_ptr<libkineto::external_api::CpuTraceBuffer> cpuTrace,
      bool logNet);

  bool inline passesGpuOpCountThreshold(
      const libkineto::external_api::CpuTraceBuffer& cpuTrace) {
    return cpuOnly_ || cpuTrace.gpuOpCount < 0 ||
        cpuTrace.gpuOpCount >= netGpuOpCountThreshold_;
  }

  // Returns true if net name is to be tracked for a specified number of
  // iterations.
  bool iterationTargetMatch(
      const libkineto::external_api::CpuTraceBuffer& trace);

  // net name to id
  int netId(const std::string& netName);

  // Process generic CUPTI activity
  void handleCuptiActivity(const CUpti_Activity* record);

  // Process specific GPU activity types
  void updateGpuNetSpan(
      uint64_t startUsecs,
      uint64_t endUsecs,
      const libkineto::external_api::OpDetails& ext);
  bool outOfRange(uint64_t startNsecs, uint64_t endNsecs);
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(const CUpti_ActivityAPI* activity);
  void handleGpuActivity(const CUpti_ActivityKernel4* kernel);
  template <class T>
  void handleGpuActivity(const T* act, const char* name);

  // Is logging disabled for this event?
  // Logging can be disabled due to operator count, net name filter etc.
  inline bool loggingDisabled(const libkineto::external_api::OpDetails& event) {
    const auto& it = opNetMap_.find(event.correlationId);
    return it != opNetMap_.end() &&
        externalDisabledNets_.find(it->second.net) !=
        externalDisabledNets_.end();
  }

  inline void recordThreadName(pthread_t pthreadId) {
    if (threadNames_.find(pthreadId) == threadNames_.end()) {
      threadNames_[pthreadId] = getThreadName(pthreadId);
    }
  }

  void resetTraceData() {
    externalEvents_.clear();
    externalDisabledNets_.clear();
    gpuNetSpanMap_.clear();
  }

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

  // Logger that dumps all the activities to a json file.
  std::unique_ptr<ActivityLogger> logger_;

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
  std::unordered_map<std::string, int> netIdMap_;

  // Maintain a map of operator to net.
  // Also keep track of which iteration of a net an operator belongs to.
  struct NetId {
    int net;
    int instance;
  };
  std::unordered_map<uint64_t, NetId> opNetMap_;
  std::vector<std::string> netNames_;
  std::unordered_map<uint64_t, std::string> threadNames_;
  // Which net ids are disabled. Together with the operator -> net id map
  // this allows us to determine whether a GPU or CUDA API event should
  // be included in the trace.
  // If a CUDA event cannot be mapped to a net it will always be included.
  std::unordered_set<int> externalDisabledNets_;
  std::unordered_map<int, std::vector<std::pair<uint64_t, uint64_t>>>
      gpuNetSpanMap_;

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

  // the queue to store CPU traces;
  // multiple producers -- transferCpuTrace
  // single consumer -- profilerLoop
  std::queue<std::pair<
      int, // instance
      std::unique_ptr<libkineto::external_api::CpuTraceBuffer>>>
      cpuTraceQueue_;
};

} // namespace KINETO_NAMESPACE
