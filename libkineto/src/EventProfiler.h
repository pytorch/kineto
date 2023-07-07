/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <cupti.h>

#include "Config.h"
#include "CuptiEventApi.h"
#include "CuptiMetricApi.h"
#include "SampleListener.h"

namespace KINETO_NAMESPACE {

// Helper function for computing percentiles (nearest-rank).
// Modifies the input.
template <typename T>
inline PercentileList& percentiles(std::vector<T> values, PercentileList& pcs) {
  auto size = values.size();
  for (auto& x : pcs) {
    int idx = std::min(size - 1, (x.first * size) / 100);
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    x.second = SampleValue(values[idx]);
  }
  return pcs;
}

// Helper function for normalizing a percentile list
// Modifies the input
inline PercentileList& normalize(PercentileList& pcs, double sf) {
  for (auto& pc : pcs) {
    pc.second *= sf;
  }
  return pcs;
}

// A slice of the sample buffer
struct SampleSlice {
  // Start offset (samples)
  int offset;
  // Slice number
  int index;
  // Out of this many
  int count;
};

// A sampled event
class Event {
 public:
  /* implicit */ Event(std::string name) : name(std::move(name)) {}
  /* implicit */ Event(const char* name) : name(name) {}
  Event() : name("INVALID") {}

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&&) = default;
  Event& operator=(Event&&) = default;

  void addSample(
      std::chrono::time_point<std::chrono::system_clock> timestamp,
      const std::vector<int64_t>& values) {
    assert(values.size() == instanceCount);
    samples_.emplace_back(timestamp, values);
  }

  // Sum samples for a single domain instance
  int64_t sumInstance(int i, const SampleSlice& slice) const;

  // Sum all samples across all domain instances
  int64_t sumAll(const SampleSlice& slice) const;

  // Create list of percentiles
  PercentileList& percentiles(PercentileList& pcs, const SampleSlice& slice)
      const;

  void eraseSamples(int count) {
    auto end = samples_.begin();
    std::advance(end, count);
    samples_.erase(samples_.begin(), end);
  }

  void clearSamples() {
    samples_.clear();
  }

  int sampleCount() {
    return samples_.size();
  }

  void printSamples(std::ostream& s, CUdevice device) const;

  // Event name (see nvprof --query-events)
  std::string name;

  // Number of domain instances for this event, e.g. number of SMs
  int instanceCount = 0;

 private:
  std::pair<int, int> toIdxRange(const SampleSlice& slice) const {
    int size = (samples_.size() - slice.offset) / slice.count;
    return std::make_pair(slice.offset + (slice.index * size), size);
  }

  // List of collected samples, where each sample has values for
  // one or more domain instances
  using Sample = std::pair<
      std::chrono::time_point<std::chrono::system_clock>,
      std::vector<int64_t>>;
  std::list<Sample> samples_;
};

class Metric {
 public:
  Metric(
      std::string name,
      CUpti_MetricID id,
      std::vector<CUpti_EventID> events,
      CUpti_MetricEvaluationMode eval_mode,
      CuptiMetricApi& cupti_metrics);

  struct CalculatedValues {
    std::vector<SampleValue> perInstance;
    SampleValue total;
  };

  struct CalculatedValues calculate(
      std::map<CUpti_EventID, Event>& events,
      std::chrono::nanoseconds sample_duration,
      const SampleSlice& slice);

  int instanceCount(std::map<CUpti_EventID, Event>& events) {
    return events[events_[0]].instanceCount;
  }

  void printDescription(std::ostream& s) const;

  std::string name;

 private:
  CUpti_MetricID id_;
  std::vector<CUpti_EventID> events_;
  CUpti_MetricEvaluationMode evalMode_;
  // Calls to CUPTI is encapsulated behind this interface
  CuptiMetricApi& cuptiMetrics_;
  CUpti_MetricValueKind valueKind_;
};

/**
 * A set of event groups.
 * Holds all the events that may be collected in a single pass.
 * A group contains one or more counters for a single domain.
 * A group set contains zero or one groups per domain.
 */
class EventGroupSet {
 public:
  EventGroupSet(
      CUpti_EventGroupSet& set,
      std::map<CUpti_EventID, Event>& events,
      CuptiEventApi& cupti);
  ~EventGroupSet();

  EventGroupSet(const EventGroupSet&) = delete;
  EventGroupSet& operator=(const EventGroupSet&) = delete;
  EventGroupSet(EventGroupSet&&) = default;
  EventGroupSet& operator=(EventGroupSet&&) = delete;

  // Number of groups = number of domains profiled
  int groupCount() const {
    return set_.numEventGroups;
  }

  void setEnabled(bool enabled);
  // Take a sample of counters in this group set
  void collectSample();
  void printDescription(std::ostream& s) const;

 private:
  CUpti_EventGroupSet& set_;
  std::map<CUpti_EventID, Event>& events_;
  // Calls to CUPTI is encapsulated behind this interface
  CuptiEventApi& cuptiEvents_;
  bool enabled_;
};

// The sampler
class EventProfiler {
 public:
  explicit EventProfiler(
      std::unique_ptr<CuptiEventApi> cupti_events,
      std::unique_ptr<CuptiMetricApi> cupti_metrics,
      std::vector<std::unique_ptr<SampleListener>>& loggers,
      std::vector<std::unique_ptr<SampleListener>>& onDemandLoggers);
  EventProfiler(const EventProfiler&) = delete;
  EventProfiler& operator=(const EventProfiler&) = delete;
  ~EventProfiler();

  void configure(Config& config, Config* onDemandConfig);

  bool isOnDemandActive() {
    return !!onDemandConfig_;
  }

  // Print the counter sets. Multiple sets will be multiplexed.
  void printSets(std::ostream& s) const;

  // Print metrics descriptions
  void printMetrics(std::ostream& s) const;

  bool enableForDevice(Config& cfg);

  CUdevice device() {
    return cuptiEvents_->device();
  }

  bool setContinuousMode() {
    return cuptiEvents_->setContinuousMode();
  }

  std::chrono::milliseconds samplePeriod() {
    return mergedConfig_->samplePeriod();
  }

  std::chrono::milliseconds multiplexPeriod() {
    return mergedConfig_->multiplexPeriod();
  }

  std::chrono::milliseconds reportPeriod() {
    return config_->reportPeriod();
  }

  std::chrono::milliseconds onDemandReportPeriod() {
    return onDemandConfig_->reportPeriod();
  }

  // Read values of currently running counters.
  void collectSample();

  void reportSamples();
  void reportOnDemandSamples();

  bool enabled() {
    return sets_.size() > 0;
  }

  bool multiplexEnabled() {
    return sets_.size() > 1;
  }

  // Multiplex counters.
  void enableNextCounterSet();

  void eraseReportedSamples() {
    int erase_count = baseSamples_;
    if (onDemandConfig_ &&
        onDemandConfig_->eventProfilerOnDemandDuration().count() > 0) {
      erase_count = std::min(baseSamples_, onDemandSamples_);
    }
    eraseSamples(erase_count);
    baseSamples_ -= erase_count;
    onDemandSamples_ -= erase_count;
  }

  void clearSamples() {
    for (auto& pair : events_) {
      pair.second.clearSamples();
    }
    baseSamples_ = 0;
    onDemandSamples_ = 0;
  }

 private:
  // Functions to initialize profiler based on Config settings.
  bool applyConfig(const Config& config);
  bool initEventsAndMetrics(const Config& config);
  void initEvents(const std::set<std::string>& eventNames);
  void initMetrics(const std::set<std::string>& metricNames);
  bool initEventGroups();

  PercentileList initPercentiles(const std::vector<int>& percentiles) {
    PercentileList res;
    res.reserve(percentiles.size());
    for (int p : percentiles) {
      res.emplace_back(p, SampleValue(0));
    }
    return res;
  }

  // Notify listeners of collected samples
  void dispatchSamples(
      const Config& config,
      const std::vector<std::unique_ptr<SampleListener>>& loggers,
      int report_nr);

  void eraseSamples(int count) {
    for (auto& pair : events_) {
      pair.second.eraseSamples(count);
    }
  }

  void updateLoggers(Config& config, Config* on_demand_config);

  // Print all collected samples since last clear.
  void printAllSamples(std::ostream& s, CUdevice device) const;

  // Calls to CUPTI is encapsulated behind these interfaces
  std::unique_ptr<CuptiEventApi> cuptiEvents_;
  std::unique_ptr<CuptiMetricApi> cuptiMetrics_;
  // The CUpti API reports event IDs, we must map them to our event objects
  std::map<CUpti_EventID, Event> events_;
  // List of metrics
  std::vector<Metric> metrics_;
  // The countert sets needed to collect all counters
  std::vector<EventGroupSet> sets_;
  // The event group set object returned by Cupti.
  // Saved s.t. we can call cuptiEventGroupSetsDestroy to free memory when
  // the object is no longer needed.
  CUpti_EventGroupSets* eventGroupSets_ = nullptr;
  // Current multiplexed counter set
  int curEnabledSet_{0};

  std::unique_ptr<Config> config_;
  std::unique_ptr<Config> onDemandConfig_;
  std::unique_ptr<Config> mergedConfig_;
  int baseSamples_{0};
  int onDemandSamples_{0};

  // Shared between profiler threads
  // Vectors are read-only but calling loggers require lock
  const std::vector<std::unique_ptr<SampleListener>>& loggers_;
  const std::vector<std::unique_ptr<SampleListener>>& onDemandLoggers_;
};

} // namespace KINETO_NAMESPACE
