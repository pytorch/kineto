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
#include "output_csv.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using std::unique_ptr;
using std::vector;

namespace KINETO_NAMESPACE {

namespace {

vector<std::function<unique_ptr<SampleListener>(const Config&)>>&
loggerFactories() {
  static vector<std::function<unique_ptr<SampleListener>(const Config&)>>
      factories;
  return factories;
}

vector<std::function<unique_ptr<SampleListener>(const Config&)>>&
onDemandLoggerFactories() {
  static vector<std::function<unique_ptr<SampleListener>(const Config&)>>
      factories;
  return factories;
}

vector<unique_ptr<SampleListener>> makeLoggers(const Config& config) {
  vector<unique_ptr<SampleListener>> loggers;
  for (const auto& factory : loggerFactories()) {
    loggers.push_back(factory(config));
  }
  loggers.push_back(std::make_unique<EventCSVDbgLogger>());
  loggers.push_back(std::make_unique<EventCSVFileLogger>());
  return loggers;
}

vector<unique_ptr<SampleListener>> makeOnDemandLoggers(
    const Config& config) {
  vector<unique_ptr<SampleListener>> loggers;
  for (const auto& factory : onDemandLoggerFactories()) {
    loggers.push_back(factory(config));
  }
  loggers.push_back(std::make_unique<EventCSVDbgLogger>());
  return loggers;
}

vector<unique_ptr<SampleListener>>& loggers(const Config& config) {
  static auto res = makeLoggers(config);
  return res;
}

vector<unique_ptr<SampleListener>>& onDemandLoggers(
    const Config& config) {
  static auto res = makeOnDemandLoggers(config);
  return res;
}

} // anon namespace

// Keep an eye on profiling threads.
// We've observed deadlocks in Cuda11 in libcuda / libcupti..
namespace detail {

class HeartbeatMonitor {

 public:
  ~HeartbeatMonitor() {
    stopMonitoring();
  }

  static HeartbeatMonitor& instance() {
    static HeartbeatMonitor monitor;
    return monitor;
  }

  void profilerHeartbeat() {
    int32_t tid = systemThreadId();
    std::lock_guard<std::mutex> lock(mutex_);
    profilerAliveMap_[tid]++;
  }

  void setPeriod(seconds period) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (period_ == period) {
        return;
      }
      period_ = period;
    }
    if (period == seconds(0)) {
      stopMonitoring();
    } else {
      startMonitoring();
    }
  }

 private:
  HeartbeatMonitor() = default;

  void monitorLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while(!stopMonitor_) {
      auto cv_status = condVar_.wait_for(lock, seconds(period_));
      // Don't perform check on spurious wakeup or on notify
      if (cv_status == std::cv_status::timeout) {
        for (auto& pair : profilerAliveMap_) {
          int32_t tid = pair.first;
          int& i = pair.second;
          if (i == 0) {
            LOG(ERROR) << "Thread " << tid << " appears stuck!";
          }
          i = 0;
        }
      }
    }
  }

  void startMonitoring() {
    if (!monitorThread_) {
      VLOG(0) << "Starting monitoring thread";
      stopMonitor_ = false;
      monitorThread_ = std::make_unique<std::thread>(
          &HeartbeatMonitor::monitorLoop, this);
    }
  }

  void stopMonitoring() {
    if (monitorThread_) {
      VLOG(0) << "Stopping monitoring thread";
      stopMonitor_ = true;
      condVar_.notify_one();
      monitorThread_->join();
      monitorThread_ = nullptr;
      VLOG(0) << "Monitoring thread terminated";
    }
  }

  std::map<int32_t, int> profilerAliveMap_;
  std::unique_ptr<std::thread> monitorThread_;
  std::mutex mutex_;
  std::condition_variable condVar_;
  std::atomic_bool stopMonitor_{false};
  seconds period_{0};
};

} // namespace detail

namespace {
// Profiler map singleton
std::map<CUcontext, unique_ptr<EventProfilerController>>& profilerMap() {
  static std::map<CUcontext, unique_ptr<EventProfilerController>> instance;
  return instance;
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

void configureHeartbeatMonitor(
    detail::HeartbeatMonitor& monitor, const Config& base, const Config* onDemand) {
  seconds base_period =
      base.eventProfilerHeartbeatMonitorPeriod();
  seconds on_demand_period = !onDemand ? seconds(0) :
      onDemand->eventProfilerHeartbeatMonitorPeriod();
  monitor.setPeriod(
      on_demand_period > seconds(0) ? on_demand_period : base_period);
}

} // anon namespace

void EventProfilerController::addLoggerFactory(
    std::function<unique_ptr<SampleListener>(const Config&)> factory) {
  loggerFactories().push_back(factory);
}

void EventProfilerController::addOnDemandLoggerFactory(
    std::function<unique_ptr<SampleListener>(const Config&)> factory) {
  onDemandLoggerFactories().push_back(factory);
}

EventProfilerController::EventProfilerController(
    CUcontext context,
    ConfigLoader& configLoader,
    detail::HeartbeatMonitor& heartbeatMonitor)
    : configLoader_(configLoader), heartbeatMonitor_(heartbeatMonitor) {
  auto cupti_events = std::make_unique<CuptiEventInterface>(context);
  auto cupti_metrics =
      std::make_unique<CuptiMetricInterface>(cupti_events->device());
  configLoader_.addHandler(
      ConfigLoader::ConfigKind::EventProfiler, this);
  auto config = configLoader.getConfigCopy();
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
  configLoader_.removeHandler(
      ConfigLoader::ConfigKind::EventProfiler, this);
  VLOG(0) << "Stopped event profiler";
}

// Must be called under lock
void EventProfilerController::start(CUcontext ctx, ConfigLoader& configLoader) {
  profilerMap()[ctx] = unique_ptr<EventProfilerController>(
      new EventProfilerController(
          ctx, configLoader, detail::HeartbeatMonitor::instance()));
}

// Must be called under lock
void EventProfilerController::stop(CUcontext ctx) {
  profilerMap()[ctx] = nullptr;
}

bool EventProfilerController::canAcceptConfig() {
  std::lock_guard<std::mutex> guard(mutex_);
  return !newOnDemandConfig_;
}

void EventProfilerController::acceptConfig(const Config& config) {
  if (config.eventProfilerOnDemandDuration().count() == 0) {
    // Ignore - not for this profiler
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  if (newOnDemandConfig_) {
    LOG(ERROR) << "On demand request already queued - ignoring new request";
    return;
  }
  newOnDemandConfig_ = config.clone();
  LOG(INFO) << "Received new on-demand config";
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

  time_point<system_clock> next_sample_time;
  time_point<system_clock> next_report_time;
  time_point<system_clock> next_on_demand_report_time;
  time_point<system_clock> next_multiplex_time;
  std::unique_ptr<Config> on_demand_config = nullptr;
  bool reconfigure = true;
  bool restart = true;
  int report_count = 0;
  int on_demand_report_count = 0;
  while (!stopRunloop_) {
    heartbeatMonitor_.profilerHeartbeat();
    if (configLoader_.hasNewConfig(*config)) {
      config = configLoader_.getConfigCopy();
      VLOG(0) << "Base config changed";
      report_count = 0;
      reconfigure = true;
    }

    auto now = system_clock::now();
    if (on_demand_config &&
        now > (on_demand_config->eventProfilerOnDemandStartTime() +
               on_demand_config->eventProfilerOnDemandDuration())) {
      on_demand_config = nullptr;
      LOG(INFO) << "On-demand profiling complete";
      reconfigure = true;
    }

    if (!profiler_->isOnDemandActive()) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (newOnDemandConfig_) {
        VLOG(0) << "Received on-demand config, reconfiguring";
        on_demand_config = std::move(newOnDemandConfig_);
        reconfigure = true;
        on_demand_report_count = 0;
      }
    }

    if (reconfigure) {
      try {
        profiler_->configure(*config, on_demand_config.get());
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Encountered error while configuring event profiler: "
            << ex.what();
        // Exit profiling entirely when encountering an error here
        // as it indicates a serious problem or bug.
        break;
      }
      configureHeartbeatMonitor(
          heartbeatMonitor_, *config, on_demand_config.get());
      reconfigure = false;
      restart = true;
    }

    if (restart) {
      now = system_clock::now();
      next_sample_time = now + profiler_->samplePeriod();
      next_report_time = now + profiler_->reportPeriod();
      if (profiler_->isOnDemandActive()) {
        next_on_demand_report_time = now + profiler_->onDemandReportPeriod();
      }
      next_multiplex_time = now + profiler_->multiplexPeriod();
      // Collect an initial sample and throw it away
      // The next sample is the first valid one
      profiler_->collectSample();
      profiler_->clearSamples();
      restart = false;
    }

    auto start_sleep = now;
    while (now < next_sample_time) {
      /* sleep override */
      std::this_thread::sleep_for(next_sample_time - now);
      now = system_clock::now();
    }
    int sleep_time = duration_cast<milliseconds>(now - start_sleep).count();

    auto start_sample = now;
    profiler_->collectSample();
    now = system_clock::now();
    int sample_time = duration_cast<milliseconds>(now - start_sample).count();

    next_sample_time += profiler_->samplePeriod();
    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, 0, 0);
      restart = true;
      continue;
    }

    auto start_report = now;
    if (now > next_report_time) {
      VLOG(1) << "Report #" << report_count++;
      profiler_->reportSamples();
      next_report_time += profiler_->reportPeriod();
    }
    if (profiler_->isOnDemandActive() && now > next_on_demand_report_time) {
      VLOG(1) << "OnDemand Report #" << on_demand_report_count++;
      profiler_->reportOnDemandSamples();
      next_on_demand_report_time += profiler_->onDemandReportPeriod();
    }
    profiler_->eraseReportedSamples();
    now = system_clock::now();
    int report_time = duration_cast<milliseconds>(now - start_report).count();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, report_time, 0);
      restart = true;
      continue;
    }

    auto start_multiplex = now;
    if (profiler_->multiplexEnabled() && now > next_multiplex_time) {
      profiler_->enableNextCounterSet();
      next_multiplex_time += profiler_->multiplexPeriod();
    }
    now = system_clock::now();
    int multiplex_time =
        duration_cast<milliseconds>(now - start_multiplex).count();

    if (now > next_sample_time) {
      reportLateSample(sleep_time, sample_time, report_time, multiplex_time);
      restart = true;
    }

    VLOG(0) << "Runloop execution time: "
            << duration_cast<milliseconds>(now - start_sample).count() << "ms";
  }

  VLOG(0) << "Device " << profiler_->device()
          << ": Exited event profiling loop";
}

} // namespace KINETO_NAMESPACE
