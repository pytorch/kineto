/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

#include <cupti.h>

namespace KINETO_NAMESPACE {

class Config;
class ConfigLoader;
class EventProfiler;
class SampleListener;

namespace detail {
class HeartbeatMonitor;
}

class EventProfilerController {
 public:
  EventProfilerController(const EventProfilerController&) = delete;
  EventProfilerController& operator=(const EventProfilerController&) = delete;

  ~EventProfilerController();

  static void start(CUcontext ctx);
  static void stop(CUcontext ctx);

  static void addLoggerFactory(
      std::function<std::unique_ptr<SampleListener>(const Config&)> factory);

  static void addOnDemandLoggerFactory(
      std::function<std::unique_ptr<SampleListener>(const Config&)> factory);

 private:
  explicit EventProfilerController(
      CUcontext context,
      ConfigLoader& configLoader,
      detail::HeartbeatMonitor& heartbeatMonitor);
  bool enableForDevice(Config& cfg);
  void profilerLoop();

  ConfigLoader& configLoader_;
  detail::HeartbeatMonitor& heartbeatMonitor_;
  std::unique_ptr<EventProfiler> profiler_;
  std::unique_ptr<std::thread> profilerThread_;
  std::atomic_bool stopRunloop_{false};
};

} // namespace KINETO_NAMESPACE
