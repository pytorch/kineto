/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "EventProfiler.h"

#include <assert.h>
#include <fmt/format.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <cupti.h>

#include "CuptiEventApi.h"
#include "Logger.h"

using namespace std::chrono;
using std::accumulate;
using std::endl;
using std::map;
using std::ostream;
using std::string;
using std::unique_ptr;
using std::vector;

namespace KINETO_NAMESPACE {

static std::mutex& logMutex() {
  static std::mutex instance;
  return instance;
}

// ---------------------------------------------------------------------
// class Event
// ---------------------------------------------------------------------

// Compute domain instance percentiles
PercentileList& Event::percentiles(
    PercentileList& pcs,
    const SampleSlice& slice) const {
  vector<int64_t> instance_values;
  instance_values.reserve(instanceCount);
  for (int i = 0; i < instanceCount; i++) {
    instance_values.push_back(sumInstance(i, slice));
  }
  return KINETO_NAMESPACE::percentiles(instance_values, pcs);
}

// Add up all samples for a given domain instance
int64_t Event::sumInstance(int i, const SampleSlice& slice) const {
  auto r = toIdxRange(slice);
  auto start = samples_.cbegin();
  std::advance(start, r.first);
  auto end = start;
  std::advance(end, r.second);
  return accumulate(start, end, 0ul, [i](int64_t a, const Sample& b) {
    return a + b.second[i];
  });
}

// Add up all samples across all domain instances
int64_t Event::sumAll(const SampleSlice& slice) const {
  int64_t res = 0;
  for (int i = 0; i < instanceCount; i++) {
    res += sumInstance(i, slice);
  }
  return res;
}

// Print raw sample values for all domains
void Event::printSamples(ostream& s, CUdevice device) const {
  // Don't mess up output with interleaved lines
  // Probably OK to reuse logMutex() here since this is
  // used for debugging, but need to keep an eye on it.
  std::lock_guard<std::mutex> lock(logMutex());
  s << "Device " << device << " " << name << ":" << endl;
  for (const auto& sample : samples_) {
    const auto& vals = sample.second;
    for (int64_t val : vals) {
      s << val << " ";
    }
    s << endl;
  }
}

// ---------------------------------------------------------------------
// class Metric
// ---------------------------------------------------------------------
Metric::Metric(
    string name,
    CUpti_MetricID id,
    vector<CUpti_EventID> events,
    CUpti_MetricEvaluationMode eval_mode,
    CuptiMetricApi& cupti_metrics)
    : name(std::move(name)),
      id_(id),
      events_(std::move(events)),
      evalMode_(eval_mode),
      cuptiMetrics_(cupti_metrics),
      valueKind_(cuptiMetrics_.valueKind(id)) {}

// Return per-SM vector as well as total
struct Metric::CalculatedValues Metric::calculate(
    map<CUpti_EventID, Event>& event_map,
    nanoseconds sample_duration,
    const SampleSlice& slice) {
  vector<SampleValue> metric_values;
  vector<int64_t> ev_values;
  ev_values.reserve(events_.size());
  if (evalMode_ & CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE) {
    int instance_count = instanceCount(event_map);
    metric_values.reserve(instance_count);
    for (int i = 0; i < instance_count; i++) {
      ev_values.clear();
      for (CUpti_EventID event_id : events_) {
        ev_values.push_back(event_map[event_id].sumInstance(i, slice));
      }
      metric_values.push_back(cuptiMetrics_.calculate(
          id_, valueKind_, events_, ev_values, sample_duration.count()));
    }
  }

  // FIXME: Check assumption that all instances are profiled
  ev_values.clear();
  for (CUpti_EventID event_id : events_) {
    ev_values.push_back(event_map[event_id].sumAll(slice));
  }
  SampleValue total = cuptiMetrics_.calculate(
      id_, valueKind_, events_, ev_values, sample_duration.count());
  if (evalMode_ & CUPTI_METRIC_EVALUATION_MODE_AGGREGATE) {
    metric_values.push_back(total);
  }
  return {metric_values, std::move(total)};
}

void Metric::printDescription(ostream& s) const {
  s << fmt::format("{} ({})", name, fmt::join(events_, ",")) << endl;
}

// ---------------------------------------------------------------------
// class EventGroupSet
// ---------------------------------------------------------------------

// Each domain has a set of counters.
// Some counters in a domain can be collected simultaneously in a "group"
// Counters from different domains can also be collected at the same time
// Therefore we have a "set of groups", or group set, with counters that
// can all be collected at once.
EventGroupSet::EventGroupSet(
    CUpti_EventGroupSet& set,
    map<CUpti_EventID, Event>& events,
    CuptiEventApi& cupti)
    : set_(set), events_(events), cuptiEvents_(cupti), enabled_(false) {
  for (int g = 0; g < set.numEventGroups; g++) {
    CUpti_EventGroup grp = set.eventGroups[g];
    // Profile all domain instances
    cuptiEvents_.enablePerInstance(grp);
    uint32_t instance_count = cuptiEvents_.instanceCount(grp);
    for (const auto& id : cuptiEvents_.eventsInGroup(grp)) {
      VLOG(0) << "Instance count for " << id << ":" << instance_count;
      events_[id].instanceCount = instance_count;
    }
  }
}

EventGroupSet::~EventGroupSet() {
  // Disable EventGroupSet in Cupti.
  if (enabled_) {
    setEnabled(false);
  }
}

// Enable or disable this group set
void EventGroupSet::setEnabled(bool enabled) {
  if (enabled && !enabled_) {
    cuptiEvents_.enableGroupSet(set_);
  } else if (!enabled && enabled_) {
    cuptiEvents_.disableGroupSet(set_);
  }
  enabled_ = enabled;
}

// Collect counter values for each counter in group set
void EventGroupSet::collectSample() {
  auto timestamp = system_clock::now();
  for (int g = 0; g < set_.numEventGroups; g++) {
    CUpti_EventGroup grp = set_.eventGroups[g];
    for (const auto& id : cuptiEvents_.eventsInGroup(grp)) {
      Event& ev = events_[id];
      vector<int64_t> vals(ev.instanceCount);
      // FIXME: Use cuptiEventGroupReadAllEvents
      cuptiEvents_.readEvent(grp, id, vals);

      if (VLOG_IS_ON(0)) {
        for (int64_t v : vals) {
          if (v == CUPTI_EVENT_OVERFLOW) {
            LOG(WARNING) << "Counter overflow detected "
                         << "- decrease sample period!" << endl;
          }
        }
      }

      ev.addSample(timestamp, vals);
    }
  }

  if (VLOG_IS_ON(1)) {
    auto t2 = system_clock::now();
    VLOG(1) << "Device " << cuptiEvents_.device() << " Sample (us): "
            << duration_cast<microseconds>(t2 - timestamp).count();
  }
}

// Print names of events in this group set, ordered by group
void EventGroupSet::printDescription(ostream& s) const {
  for (int g = 0; g < set_.numEventGroups; g++) {
    s << "  Events in group " << g << ": ";
    for (const auto& id : cuptiEvents_.eventsInGroup(set_.eventGroups[g])) {
      s << id << " (" << events_[id].name << ") ";
    }
    s << endl;
  }
}

// ---------------------------------------------------------------------
// class EventProfiler
// ---------------------------------------------------------------------

// Find nearest factor of a number by linear search,
// starting at hi and lo - hi searches up and lo searches down
static int nearestFactor(int hi, int lo, int number) {
  return number % hi == 0
      ? hi
      : number % lo == 0 ? lo : nearestFactor(hi + 1, lo - 1, number);
}

static int nearestFactor(int count, int max) {
  return nearestFactor(count, count, max);
}

void EventProfiler::initEvents(const std::set<std::string>& eventNames) {
  events_.clear();
  // Build event map
  for (const auto& name : eventNames) {
    events_.emplace(cuptiEvents_->eventId(name), name);
  }
}

void EventProfiler::initMetrics(const std::set<std::string>& metricNames) {
  metrics_.clear();
  // Add events from metrics
  metrics_.reserve(metricNames.size());
  for (const auto& metric_name : metricNames) {
    CUpti_MetricID metric_id = cuptiMetrics_->idFromName(metric_name);
    if (metric_id == ~0) {
      continue;
    }

    const auto& events = cuptiMetrics_->events(metric_id);
    vector<CUpti_EventID> event_ids;
    event_ids.reserve(events.size());
    for (const auto& pair : events) {
      CUpti_EventID id = pair.first;
      const string& event_name = pair.second;
      if (event_name.empty()) {
        // For unnamed events, use metric name and event id
        // FIXME: For subsequent metrics using the same event,
        // this will be confusing
        events_.emplace(id, metric_name + "_" + event_name);
      } else {
        events_.emplace(id, event_name);
      }
      event_ids.push_back(id);
    }
    metrics_.emplace_back(
        metric_name,
        metric_id,
        event_ids,
        cuptiMetrics_->evaluationMode(metric_id),
        *cuptiMetrics_);
  }
}

bool EventProfiler::initEventGroups() {
  sets_.clear();
  if (eventGroupSets_) {
    cuptiEvents_->destroyGroupSets(eventGroupSets_);
    eventGroupSets_ = nullptr;
  }
  if (events_.empty()) {
    return true;
  }

  // Determine sets of groups to be collected
  vector<CUpti_EventID> ids;
  ids.reserve(events_.size());
  for (const auto& ev : events_) {
    ids.push_back(ev.first);
  }
  eventGroupSets_ = cuptiEvents_->createGroupSets(ids);
  VLOG(0) << "Number of group sets: " << eventGroupSets_->numSets;
  for (int i = 0; i < eventGroupSets_->numSets; i++) {
    sets_.push_back(
        EventGroupSet(eventGroupSets_->sets[i], events_, *cuptiEvents_));
  }
  return !sets_.empty();
}

static unique_ptr<Config> alignAndValidateConfigs(
    Config& base,
    Config* onDemand) {
  auto now = system_clock::now();
  if (!onDemand ||
      now >
          (onDemand->eventProfilerOnDemandStartTime() +
           onDemand->eventProfilerOnDemandDuration())) {
    base.validate(now);
    return base.clone();
  }

  auto res = base.clone();
  res->addEvents(onDemand->eventNames());
  res->addMetrics(onDemand->metricNames());

  int sample_period =
      std::min(base.samplePeriod().count(), onDemand->samplePeriod().count());
  if (sample_period < base.samplePeriod().count() &&
      (base.samplePeriod().count() % sample_period) != 0) {
    sample_period = nearestFactor(sample_period, base.samplePeriod().count());
    LOG(WARNING)
        << "On-demand sample period must be a factor of base sample period. "
        << "Adjusting from " << onDemand->samplePeriod().count() << "ms to "
        << sample_period << "ms.";
  }
  base.setSamplePeriod(milliseconds(sample_period));
  base.validate(now);
  res->setSamplePeriod(base.samplePeriod());
  res->setMultiplexPeriod(base.multiplexPeriod());
  res->validate(now);
  onDemand->setSamplePeriod(base.samplePeriod());
  onDemand->setMultiplexPeriod(base.multiplexPeriod());
  onDemand->validate(now);

  return res;
}

static milliseconds minReportPeriod(const Config& config, int num_sets) {
  return config.multiplexPeriod() * num_sets;
}

static bool canSupportReportPeriod(const Config& config, int num_sets) {
  // Can we get through the groups an even number per report period?
  milliseconds min_report_period = minReportPeriod(config, num_sets);
  return (config.reportPeriod().count() % min_report_period.count()) == 0;
}

static int completeSamplesPerReport(const Config& config, int num_sets) {
  if (num_sets <= 1) {
    return config.reportPeriod() / config.samplePeriod();
  }
  // Numnber of complete sample collections in the report period
  // E.g. if report period is 10000ms, sample period 500ms,
  // multiplex period 2000ms and num_sets is 5 then # of complete samples is
  // (2000ms / 500ms) * (10000ms / 2000ms / 5) = 4 * 1 = 4
  int samples_per_multiplex_period =
      config.multiplexPeriod() / config.samplePeriod();
  int multiplex_periods_per_report =
      config.reportPeriod() / config.multiplexPeriod();
  return (multiplex_periods_per_report / num_sets) *
      samples_per_multiplex_period;
}

static bool canSupportSamplesPerReport(const Config& config, int num_sets) {
  // Can samples per report can be honored with an exact *full* set of samples?
  // We don't support partial samples at this point.
  int full_samples_per_report = completeSamplesPerReport(config, num_sets);
  return (full_samples_per_report % config.samplesPerReport()) == 0;
}

static void adjustConfig(Config& config, int num_sets) {
  // Don't change sample period and multiplex period here, since that can
  // cause overflows and perf degradation. Report period and samples per
  // report is OK to change (with warning).
  if (!canSupportReportPeriod(config, num_sets)) {
    milliseconds min_report_period = minReportPeriod(config, num_sets);
    LOG(WARNING) << "Report period must be a multiple of "
                 << min_report_period.count() << "ms (" << num_sets
                 << " event sets * " << config.multiplexPeriod().count()
                 << "ms multiplex period), in order to get complete samples.";
    auto new_report_period =
        Config::alignUp(config.reportPeriod(), min_report_period);
    double sf =
        ((double)new_report_period.count()) / config.reportPeriod().count();
    int new_samples_per_report = std::round(config.samplesPerReport() * sf);
    LOG(WARNING) << "Adjusting report period from "
                 << config.reportPeriod().count() << "ms to "
                 << new_report_period.count() << "ms";
    if (new_samples_per_report != config.samplesPerReport()) {
      LOG(WARNING) << "Adjusting samples per report from "
                   << config.samplesPerReport() << " to "
                   << new_samples_per_report;
    }
    config.setReportPeriod(new_report_period);
    config.setSamplesPerReport(new_samples_per_report);
  }
  // Ensure that samples per report can be honored with
  // an exact *full* set of samples. Don't support partial
  // samples at this point.
  if (!canSupportSamplesPerReport(config, num_sets)) {
    int full_samples_per_report = completeSamplesPerReport(config, num_sets);
    int adjusted_count =
        nearestFactor(config.samplesPerReport(), full_samples_per_report);
    LOG(WARNING)
        << "Samples per report must be such that an even number of "
        << "complete samples can be aggregated in each report period. Adjusting"
        << " from " << config.samplesPerReport() << " to " << adjusted_count
        << " (complete sample count is " << full_samples_per_report << ")";
    config.setSamplesPerReport(adjusted_count);
  }
}

// Prepare profiler
EventProfiler::EventProfiler(
    std::unique_ptr<CuptiEventApi> cupti_events,
    std::unique_ptr<CuptiMetricApi> cupti_metrics,
    vector<unique_ptr<SampleListener>>& loggers,
    vector<unique_ptr<SampleListener>>& onDemandLoggers)
    : cuptiEvents_(std::move(cupti_events)),
      cuptiMetrics_(std::move(cupti_metrics)),
      loggers_(loggers),
      onDemandLoggers_(onDemandLoggers) {}

void EventProfiler::reportSamples() {
  dispatchSamples(*config_, loggers_, baseSamples_);
  baseSamples_ += completeSamplesPerReport(*config_, sets_.size());
}

void EventProfiler::reportOnDemandSamples() {
  dispatchSamples(*onDemandConfig_, onDemandLoggers_, onDemandSamples_);
  onDemandSamples_ += completeSamplesPerReport(*onDemandConfig_, sets_.size());
}

EventProfiler::~EventProfiler() {
  if (eventGroupSets_) {
    for (auto& set : sets_) {
      set.setEnabled(false);
    }
    cuptiEvents_->destroyGroupSets(eventGroupSets_);
  }
  VLOG(0) << "Stopped event profiler for device " << device();
}

void EventProfiler::updateLoggers(Config& config, Config* on_demand_config) {
  // Update loggers.
  for (auto& logger : loggers_) {
    std::lock_guard<std::mutex> lock(logMutex());
    logger->update(config);
  }

  if (on_demand_config) {
    // Update onDemand loggers.
    for (auto& logger : onDemandLoggers_) {
      std::lock_guard<std::mutex> lock(logMutex());
      logger->update(*on_demand_config);
    }
  }
}

bool EventProfiler::applyConfig(const Config& config) {
  // Initialize events, metrics, and event group sets.
  // TODO: Send warnings / errors back to dyno for onDemand config
  try {
    if (!initEventsAndMetrics(config)) {
      return false;
    }
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Failed to apply config (" << ex.what() << ")";
    return false;
  }

  return true;
}

bool EventProfiler::initEventsAndMetrics(const Config& config) {
  initEvents(config.eventNames());
  initMetrics(config.metricNames());
  // We now have the total list of events to collect
  // They need to be organized into groups for multiplexing
  if (!initEventGroups()) {
    LOG(WARNING) << "No events/metrics initialized successfully";
    return false;
  }

  if (VLOG_IS_ON(1)) {
    printMetrics(LIBKINETO_DBG_STREAM);
    printSets(LIBKINETO_DBG_STREAM);
  }
  return true;
}

void EventProfiler::printSets(ostream& s) const {
  for (int i = 0; i < sets_.size(); i++) {
    s << "Set " << i << endl;
    sets_[i].printDescription(s);
  }
}

void EventProfiler::printMetrics(ostream& s) const {
  s << "Metrics:" << endl;
  for (const Metric& m : metrics_) {
    m.printDescription(s);
  }
}

void EventProfiler::printAllSamples(ostream& s, CUdevice device) const {
  for (const auto& pair : events_) {
    const Event& ev = pair.second;
    ev.printSamples(s, device);
  }
}

void EventProfiler::enableNextCounterSet() {
  if (sets_.size() > 1) {
    auto t1 = system_clock::now();

    VLOG(1) << "Disabling set " << curEnabledSet_;
    sets_[curEnabledSet_].setEnabled(false);
    curEnabledSet_ = (curEnabledSet_ + 1) % sets_.size();
    VLOG(1) << "Enabling set " << curEnabledSet_;
    sets_[curEnabledSet_].setEnabled(true);

    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      VLOG(1) << "Switch (us): "
              << duration_cast<microseconds>(t2 - t1).count();
    }
  }
}

// Notify listeners of collected samples
void EventProfiler::dispatchSamples(
    const Config& config,
    const vector<unique_ptr<SampleListener>>& loggers,
    int sample_offset) {
  Sample sample(events_.size() + metrics_.size());
  // Normalize values to per second
  auto delta = config.reportPeriod() / config.samplesPerReport();
  double sf = 1000.0 * sets_.size() / delta.count();
  for (int i = 0; i < config.samplesPerReport(); i++) {
    sample.stats.clear();
    sample.deltaMsec = (delta * i).count();
    SampleSlice slice = {sample_offset, i, config.samplesPerReport()};
    VLOG(1) << "Slice: " << sample_offset << ", " << i << ", "
            << config.samplesPerReport();
    for (const auto& pair : events_) {
      const Event& ev = pair.second;
      int64_t total = std::round(sf * ev.sumAll(slice));
      PercentileList pcs = initPercentiles(config.percentiles());
      normalize(ev.percentiles(pcs, slice), sf);
      sample.stats.push_back({ev.name, std::move(pcs), SampleValue(total)});
    }

    for (auto& m : metrics_) {
      // calculate returns a pair of per-SM vector and a total
      auto vals = m.calculate(events_, delta, slice);
      PercentileList pcs = initPercentiles(config.percentiles());
      sample.stats.push_back(
          {m.name, std::move(percentiles(vals.perInstance, pcs)), vals.total});
    }

    for (auto& logger : loggers) {
      std::lock_guard<std::mutex> lock(logMutex());
      logger->handleSample(device(), sample, config.ipcFabricEnabled());
    }
  }

  if (VLOG_IS_ON(2)) {
    printAllSamples(LIBKINETO_DBG_STREAM, device());
  }
}

void EventProfiler::configure(Config& config, Config* onDemandConfig) {
  if (!sets_.empty()) {
    sets_[curEnabledSet_].setEnabled(false);
    clearSamples();
  }

  config_ = config.clone();
  onDemandConfig_ = onDemandConfig ? onDemandConfig->clone() : nullptr;
  mergedConfig_ = alignAndValidateConfigs(*config_, onDemandConfig_.get());
  if (!applyConfig(*mergedConfig_)) {
    LOG(WARNING) << "Failed to apply config!";
    mergedConfig_ = config_->clone();
    applyConfig(*config_);
  }
  if (!sets_.empty()) {
    // Make timing adjustments based on multiplexing requirements.
    adjustConfig(*config_, sets_.size());
    if (onDemandConfig_) {
      int duration = onDemandConfig_->eventProfilerOnDemandDuration().count();
      LOG(INFO) << "On demand profiler activated for " << duration << " secs";
      adjustConfig(*onDemandConfig_, sets_.size());
    }
    // If events or metrics were added or removed, need to tell loggers
    updateLoggers(*config_, onDemandConfig_.get());
  }

  curEnabledSet_ = 0;
  if (!sets_.empty()) {
    sets_[0].setEnabled(true);
  } else {
    VLOG(0) << "No counters profiled!";
  }

  baseSamples_ = 0;
  onDemandSamples_ = 0;
}

void EventProfiler::collectSample() {
  if (sets_.empty()) {
    return;
  }
  sets_[curEnabledSet_].collectSample();
  if (VLOG_IS_ON(1)) {
    printAllSamples(LIBKINETO_DBG_STREAM, device());
  }
}

} // namespace KINETO_NAMESPACE
