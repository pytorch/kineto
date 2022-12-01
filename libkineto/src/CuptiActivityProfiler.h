/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude

#ifdef HAS_CUPTI
#include <cupti.h>
#include "CuptiActivity.h"
#endif // HAS_CUPTI

#include "libkineto.h"
#include "output_base.h"
#include "output_membuf.h"
#include "time_since_epoch.h"
#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ICompositeProfiler.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

namespace KINETO_NAMESPACE {

class Config;
class CuptiActivityApi;
class RoctracerActivityApi;

class CuptiActivityProfilerSession : public IActivityProfilerSession {
 public:
  explicit CuptiActivityProfilerSession(
#ifdef HAS_ROCTRACER
      RoctracerActivityApi& cupti,
#else
      CuptiActivityApi& cupti,
#endif
      const Config& config,
      bool cpuOnly,
      ICompositeProfilerSession* parentSession) :
      setupOverhead{0, 0},
      flushOverhead{0, 0},
      cupti_(cupti),
      config_(config.clone()),
      cpuOnly_(cpuOnly),
      parent_(parentSession) {
    traceBuffers_ = std::make_unique<ActivityBuffers>();
  }

  std::mutex& mutex() override {
    return mutex_;
  }

  TraceStatus status() override {
    return status_;
  }

  void status(TraceStatus status) {
    if (status_ != TraceStatus::ERROR) {
      status_ = status;
    }
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void log(ActivityLogger& logger) override;

  void start() {
    if (status_ != TraceStatus::WARMUP) {
      status_ = TraceStatus::ERROR;
      return;
    }
    status_ = TraceStatus::RECORDING;
  }

  void stop() {
    if (status_ != TraceStatus::WARMUP &&
        status_ != TraceStatus::RECORDING) {
      status_ = TraceStatus::ERROR;
      return;
    }
    status_ = TraceStatus::PROCESSING;
  }

  // Registered with client API to pass CPU trace events over
  void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) override;

  // data structure to collect cuptiActivityFlushAll() latency overhead
  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

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

  // the overhead to enable/disable activity tracking
  profilerOverhead setupOverhead;

  // the overhead to flush the activity buffer
  profilerOverhead flushOverhead;

 private:

  using CpuGpuSpanPair = std::pair<TraceSpan, TraceSpan>;

  // Map of gpu activities to user defined events
  class GpuUserEventMap {
   public:
    // Insert a user defined event which maps to the gpu trace activity.
    // If the user defined event mapping already exists this will update the
    // gpu side span to include the span of gpuTraceActivity.
    void insertOrExtendEvent(const ITraceActivity& cpuTraceActivity,
      const ITraceActivity& gpuTraceActivity);
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

  void processTrace(ActivityLogger& logger);

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  // Process a single CPU trace
  void processCpuTrace(
      libkineto::CpuTraceBuffer& cpuTrace,
      ActivityLogger& logger);

  // Create resource names for streams
  inline void recordStream(int device, int id, const std::string& postfix) {
    if (parent_) {
      parent_->recordResourceInfo(device, id, id, [id, postfix](){
         return fmt::format("stream {} {}", std::abs(id), postfix);
      });
    }
  }

  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);


  // Record client trace span for subsequent lookups from activities
  // Also creates a corresponding GPU-side span.
  CpuGpuSpanPair& recordTraceSpan(TraceSpan& span, int gpuOpCount);

#ifdef HAS_CUPTI
  // Process generic CUPTI activity
  void handleCuptiActivity(const CUpti_Activity* record, ActivityLogger* logger);

  // Process specific GPU activity types
  void updateGpuNetSpan(const ITraceActivity& gpuOp);
  bool outOfRange(const ITraceActivity& act);
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity, ActivityLogger* logger);
  void handleOverheadActivity(
      const CUpti_ActivityOverhead* activity, ActivityLogger* logger);
  void handleGpuActivity(const ITraceActivity& act,
      ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);
#endif // HAS_CUPTI

  void checkTimestampOrder(const ITraceActivity* act1);

  GpuUserEventMap gpuUserEventMap_;
  // id -> activity*
  std::unordered_map<int64_t, const ITraceActivity*> activityMap_;
  // cuda runtime id -> pytorch op id
  // CUPTI provides a mechanism for correlating Cuda events to arbitrary
  // external events, e.g.operator activities from PyTorch.
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  // CUDA runtime <-> GPU Activity
  std::unordered_map<int64_t, const ITraceActivity*>
      correlatedCudaActivities_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;

  // Mutex to protect non-atomic access to below state
  std::mutex mutex_;

  // Calls to CUPTI is encapsulated behind this interface
#ifdef HAS_ROCTRACER
  RoctracerActivityApi& cupti_;		// Design failure here
#else
  CuptiActivityApi& cupti_;
#endif

  std::unique_ptr<Config> config_;
  bool cpuOnly_{false};

  TraceStatus status_{TraceStatus::READY};

  // span name -> iteration count
  std::map<std::string, int> iterationCountMap_;
  // Buffers where trace data is stored
  std::unique_ptr<ActivityBuffers> traceBuffers_;

  // Parent session
  ICompositeProfilerSession* parent_{nullptr};


  // All recorded trace spans, both CPU and GPU
  // Trace Id -> list of iterations.
  // Using map of lists for the iterator semantics, since we are recording
  // pointers to the elements in this structure.
  std::map<std::string, std::list<CpuGpuSpanPair>> traceSpans_;

  // Maintain a map of client trace activity to trace span.
  // Maps correlation id -> TraceSpan* held by traceSpans_.
  using ActivityTraceMap = std::unordered_map<int64_t, CpuGpuSpanPair*>;
  ActivityTraceMap clientActivityTraceMap_;

};

class CuptiActivityProfiler : public IActivityProfiler {
 public:
  CuptiActivityProfiler(
      const std::string& name,
      CuptiActivityApi& cupti,
      bool cpuOnly);
  CuptiActivityProfiler(const CuptiActivityProfiler&) = delete;
  CuptiActivityProfiler& operator=(const CuptiActivityProfiler&) = delete;

  const std::string name() const override {
    return name_;
  }

  // returns activity types this profiler supports
  const std::set<ActivityType>& supportedActivityTypes() const override;

  void init(ICompositeProfiler* parent) override {}

  bool isInitialized() const override {
    return true;
  }

  bool isActive() const override {
    return session_ != nullptr;
  }

  std::shared_ptr<IActivityProfilerSession> configure(
      const Config& config,
      ICompositeProfilerSession* parentSession) override;

  void start(IActivityProfilerSession& session) override;
  void stop(IActivityProfilerSession& session) override;

 private:


  // Logger used during trace processing
  //ActivityLogger* logger_;
  //std::unique_ptr<MemoryTraceLogger> memLogger_;
  //
  const std::string name_;

  // Calls to CUPTI is encapsulated behind this interface
  CuptiActivityApi& cupti_;

  std::shared_ptr<CuptiActivityProfilerSession> session_;

  bool cpuOnly_{false};
};

} // namespace KINETO_NAMESPACE
