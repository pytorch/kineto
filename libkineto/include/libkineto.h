/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mediator for initialization and profiler control

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>

#include "IActivityProfiler.h"
#include "ThreadUtil.h"
#include "ICompositeProfiler.h"

extern "C" {
  void suppressLibkinetoLogMessages();
  int InitializeInjection(void);
  void libkineto_init(bool cpuOnly, bool logOnError);
}

namespace libkineto {

class Config;
class ConfigLoader;

using ChildActivityProfilerFactory =
  std::function<std::unique_ptr<IActivityProfiler>()>;

class LibkinetoApi {
 public:

  explicit LibkinetoApi(
      ConfigLoader& configLoader,
      std::unique_ptr<ICompositeProfiler> activityProfiler)
      : configLoader_(configLoader),
        activityProfiler_(std::move(activityProfiler)) {
  }

  ICompositeProfiler& activityProfiler() {
    return *activityProfiler_;
  }

  void suppressLogMessages() {
    suppressLibkinetoLogMessages();
  }

  // Provides access to profiler configuration management
  ConfigLoader& configLoader() {
    return configLoader_;
  }

 private:

  ConfigLoader& configLoader_;
  std::unique_ptr<ICompositeProfiler> activityProfiler_;
};

// Singleton
LibkinetoApi& api();

} // namespace libkineto
