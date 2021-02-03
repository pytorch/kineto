/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "EventProfilerController.h"

#include <chrono>
#include <thread>
#include <vector>

#include "ConfigLoader.h"
#include "CuptiEventInterface.h"
#include "CuptiMetricInterface.h"
#include "EventProfiler.h"
#include "ThreadName.h"
#include "output_csv.h"

#include "Logger.h"

using namespace std::chrono;
using std::unique_ptr;
using std::vector;

namespace KINETO_NAMESPACE {

static vector<std::function<unique_ptr<SampleListener>(const Config&)>>&
loggerFactories() {
  static vector<std::function<unique_ptr<SampleListener>(const Config&)>>
      factories;
  return factories;
}

void EventProfilerController::addLoggerFactory(
    std::function<unique_ptr<SampleListener>(const Config&)> factory) {
  loggerFactories().push_back(factory);
}

static vector<std::function<unique_ptr<SampleListener>(const Config&)>>&
onDemandLoggerFactories() {
  static vector<std::function<unique_ptr<SampleListener>(const Config&)>>
      factories;
  return factories;
}

void EventProfilerController::addOnDemandLoggerFactory(
    std::function<unique_ptr<SampleListener>(const Config&)> factory) {
  onDemandLoggerFactories().push_back(factory);
}

static vector<unique_ptr<SampleListener>> makeLoggers(const Config& config) {
  vector<unique_ptr<SampleListener>> loggers;
  for (const auto& factory : loggerFactories()) {
    loggers.push_back(factory(config));
  }
  loggers.push_back(std::make_unique<EventCSVDbgLogger>());
  loggers.push_back(std::make_unique<EventCSVFileLogger>());
  return loggers;
}

static vector<unique_ptr<SampleListener>> makeOnDemandLoggers(
    const Config& config) {
  vector<unique_ptr<SampleListener>> loggers;
  for (const auto& factory : onDemandLoggerFactories()) {
    loggers.push_back(factory(config));
  }
  loggers.push_back(std::make_unique<EventCSVDbgLogger>());
  return loggers;
}

static vector<unique_ptr<SampleListener>>& loggers(const Config& config) {
  static auto res = makeLoggers(config);
  return res;
}

static vector<unique_ptr<SampleListener>>& onDemandLoggers(
    const Config& config) {
  static auto res = makeOnDemandLoggers(config);
  return res;
}

EventProfilerController::EventProfilerController(
    CUcontext context,
    ConfigLoader& config_loader)
    : configLoader_(config_loader) {
  auto cupti_events = std::make_unique<CuptiEventInterface>(context);
  auto cupti_metrics =
      std::make_unique<CuptiMetricInterface>(cupti_events->device());
  auto config = config_loader.getConfigCopy();
  profiler_ = std::make_unique<EventProfiler>(
      std::move(cupti_events),
      std::move(cupti_metrics),
      loggers(*config),
      onDemandLoggers(*config));
  profilerThread_ = std::make_unique<std::thread>(
      &EventProfilerController::profilerLoop, this);
}

EventProfilerController::~EventProfilerController() {
  if (profilerThread_) {
    // signaling termination of the profiler loop
    stopRunloop_ = true;
    profilerThread_->join();
  }
  VLOG(0) << "Stopped event profiler";
}

// Profiler map singleton
static std::map<CUcontext, unique_ptr<EventProfilerController>>& profilerMap() {
  static std::map<CUcontext, unique_ptr<EventProfilerController>> instance;
  return instance;
}

// Must be called under lock
void EventProfilerController::start(CUcontext ctx) {
  profilerMap()[ctx] = unique_ptr<EventProfilerController>(
      new EventProfilerController(ctx, ConfigLoader::instance()));
}

// Must be called under lock
void EventProfilerController::stop(CUcontext ctx) {
  profilerMap()[ctx] = nullptr;
}

bool EventProfilerController::enableForDevice(Config& cfg) {
  // FIXME: Use device unique id!
  if (!cfg.eventProfilerEnabledForDevice(profiler_->device())) {
    return false;
  }
  // context count includes the new context
  int instances = configLoader_.contextCountForGpu(profiler_->device());
  VLOG(0) << "Device context count: " << instances;
  return instances >= 0 && instances <= cfg.maxEventProfilersPerGpu();
}

void reportLateSample(
    int sleepMs,
    int sampleMs,
    int reportMs,
    int reprogramMs) {
  LOG_EVERY_N(WARNING, 10) << "Lost sample due to delays (ms): " << sleepMs
                           << ", " << sampleMs << ", " << reportMs << ", "
                           << reprogramMs;
}

void EventProfilerController::profilerLoop() {
  // We limit the number of profilers that can exist per GPU
  auto config = configLoader_.getConfigCopy();
  if (!enableForDevice(*config)) {
    VLOG(0) << "Not starting EventProfiler - profilers for GPU "
            << profiler_->device() << " exceeds profilers per GPU limit ("
            << config->maxEventProfilersPerGpu() << ")";
    return;
  }

  if (!profiler_->setContinuousMode()) {
    VLOG(0) << "Continuous mode not supported for GPU "
            << profiler_->device() << ". Not starting Event Profiler.";
    return;
  }

  VLOG(0) << "Starting Event Profiler for GPU " << profiler_->device();
  setThreadName("CUPTI Event Profiler");

  auto on_demand_config = std::make_unique<Config>();

  time_point<high_resolution_clock> next_sample_time;
  time_point<high_resolution_clock> next_report_time;
  time_point<high_resolution_clock> next_on_demand_report_time;
  time_point<high_resolution_clock> next_multiplex_time;
  bool reconfigure = true;
  int report_count = 0;
  int on_demand_report_count = 0;
  while (!stopRunloop_) {
    if (configLoader_.hasNewConfig(*config)) {
      config = configLoader_.getConfigCopy();
      VLOG(0) << "Base config changed";
      report_count = 0;
      reconfigure = true;
    }
    if (configLoader_.hasNewEventProfilerOnDemandConfig(*on_demand_config)) {
      on_demand_config = configLoader_.getEventProfilerOnDemandConfigCopy();
      LOG(INFO) << "Received new on-demand config";
      on_demand_report_count = 0;
      reconfigure = true;
    }

    auto now = high_resolution_clock::now();
    if (on_demand_config->eventProfilerOnDemandDuration().count() > 0 &&
        now > (on_demand_config->eventProfilerOnDemandStartTime() +
               on_demand_config->eventProfilerOnDemandDuration())) {
      on_demand_config->setEventProfilerOnDemandDuration(seconds(0));
      LOG(INFO) << "On-demand profiling complete";
      reconfigure = true;
    }

    if (reconfigure) {
      try {
        profiler_->configure(*config, *on_demand_config);
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Encountered error while configuring event profiler: "
            << ex.what();
        // Exit profiling entirely when encountering an error here
        // as it indicates a serious problem or bug.
        break;
      }
      now = high_resolution_clock::now();
      next_sample_time = now + profiler_->samplePeriod();
      next_report_time = now + profiler_->reportPeriod();
      next_on_demand_report_time = now + profiler_->onDemandReportPeriod();
      next_multiplex_time = now + profiler_->multiplexPeriod();
      reconfigure = false;
    }
    auto start_sleep = now;
    while (now < next_sample_time) {
      /* sleep override */
      std::this_thread::sleep_for(next_sample_time - now);
      now = high_resolution_clock::now();
    }
    int sleep_time = duration_cast<milliseconds>(now - start_sleep).count();

    next_sample_time += profiler_->samplePeriod();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, 0, 0, 0);
      reconfigure = true;
      continue;
    }

    auto start_sample = now;
    profiler_->collectSample();
    now = high_resolution_clock::now();
    int sample_time = duration_cast<milliseconds>(now - start_sample).count();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, 0, 0);
      reconfigure = true;
      continue;
    }

    auto start_report = now;
    if (now > next_report_time) {
      VLOG(1) << "Report #" << report_count++;
      profiler_->reportSamples();
      next_report_time += profiler_->reportPeriod();
    }
    if (on_demand_config->eventProfilerOnDemandDuration().count() > 0 &&
        now > next_on_demand_report_time) {
      VLOG(1) << "OnDemand Report #" << on_demand_report_count++;
      profiler_->reportOnDemandSamples();
      next_on_demand_report_time += profiler_->onDemandReportPeriod();
    }
    profiler_->eraseReportedSamples();
    now = high_resolution_clock::now();
    int report_time = duration_cast<milliseconds>(now - start_report).count();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, report_time, 0);
      reconfigure = true;
      continue;
    }

    auto start_multiplex = now;
    if (profiler_->multiplexEnabled() && now > next_multiplex_time) {
      profiler_->enableNextCounterSet();
      next_multiplex_time += profiler_->multiplexPeriod();
    }
    now = high_resolution_clock::now();
    int multiplex_time =
        duration_cast<milliseconds>(now - start_multiplex).count();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, report_time, multiplex_time);
      reconfigure = true;
      continue;
    }

    VLOG(0) << "Runloop execution time: "
            << duration_cast<milliseconds>(now - start_sample).count() << "ms";
  }

  VLOG(0) << "Device " << profiler_->device()
          << ": Exited event profiling loop";
}

} // namespace KINETO_NAMESPACE
