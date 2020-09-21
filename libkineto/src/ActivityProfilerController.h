/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "ActivityProfiler.h"

namespace KINETO_NAMESPACE {

class Config;
class ConfigLoader;

using ActivityLoggerFactory =
    std::function<std::unique_ptr<ActivityLogger>(const Config&)>;

class ActivityProfilerController {
 public:
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController();

  static void init(bool cpuOnly);
  static void setLoggerFactory(const ActivityLoggerFactory& factory);

 private:
  explicit ActivityProfilerController(
      ConfigLoader& config_loader,
      bool cpuOnly);
  void profilerLoop();

  std::unique_ptr<ActivityProfilerController> activityProfilerController_;
  ConfigLoader& configLoader_;
  ActivityProfiler profiler_;
  std::thread* profilerThread_;
  std::atomic_bool stopRunloop_{false};
};

} // namespace KINETO_NAMESPACE
