/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <cupti.h>

#include "ConfigLoader.h"

namespace KINETO_NAMESPACE {

class Config;
class ConfigLoader;
class EventProfiler;
class SampleListener;

namespace detail {
class HeartbeatMonitor;
}

class EventProfilerController : public ConfigLoader::ConfigHandler {
 public:
  EventProfilerController(const EventProfilerController&) = delete;
  EventProfilerController& operator=(const EventProfilerController&) = delete;

  ~EventProfilerController();

  static void start(CUcontext ctx, ConfigLoader& configLoader);
  static void stopIfEnabled(CUcontext ctx);

  static void addLoggerFactory(
      std::function<std::unique_ptr<SampleListener>(const Config&)> factory);

  static void addOnDemandLoggerFactory(
      std::function<std::unique_ptr<SampleListener>(const Config&)> factory);

  bool canAcceptConfig() override;

  void acceptConfig(const Config& config) override;

 private:
  explicit EventProfilerController(
      CUcontext context,
      ConfigLoader& configLoader,
      detail::HeartbeatMonitor& heartbeatMonitor);
  bool enableForDevice(Config& cfg);
  void profilerLoop();
  static bool& started();

  ConfigLoader& configLoader_;
  std::unique_ptr<Config> newOnDemandConfig_;
  detail::HeartbeatMonitor& heartbeatMonitor_;
  std::unique_ptr<EventProfiler> profiler_;
  std::unique_ptr<std::thread> profilerThread_;
  std::atomic_bool stopRunloop_{false};
  std::mutex mutex_;
};

} // namespace KINETO_NAMESPACE
