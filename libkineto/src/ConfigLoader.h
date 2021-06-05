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
#include <mutex>
#include <string>
#include <thread>

#include "Config.h"

namespace libkineto {
  class LibkinetoApi;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
class DaemonConfigLoader;

class ConfigLoader {
 public:
  static ConfigLoader& instance();

  inline std::unique_ptr<Config> getConfigCopy() {
    std::lock_guard<std::mutex> lock(configLock_);
    return config_.clone();
  }

  inline std::unique_ptr<Config> getEventProfilerOnDemandConfigCopy() {
    std::lock_guard<std::mutex> lock(configLock_);
    return onDemandEventProfilerConfig_->clone();
  }

  bool hasNewConfig(const Config& oldConfig);
  bool hasNewEventProfilerOnDemandConfig(const Config& oldConfig);
  int contextCountForGpu(uint32_t gpu);

  void handleOnDemandSignal();

  static void setDaemonConfigLoaderFactory(
      std::function<std::unique_ptr<DaemonConfigLoader>()> factory);

 private:
  explicit ConfigLoader(libkineto::LibkinetoApi& api);
  ~ConfigLoader();

  void updateConfigThread();
  void updateBaseConfig();

  // Create configuration when receiving SIGUSR2
  void configureFromSignal(
      std::chrono::time_point<std::chrono::system_clock> now,
      Config& config);

  // Create configuration when receiving request from a daemon
  void configureFromDaemon(
      std::chrono::time_point<std::chrono::system_clock> now,
      Config& config);

  inline bool eventProfilerRequest(const Config& config) {
    return (
        config.eventProfilerOnDemandStartTime() >
        onDemandEventProfilerConfig_->eventProfilerOnDemandStartTime());
  }

  std::string readOnDemandConfigFromDaemon(
      std::chrono::time_point<std::chrono::system_clock> now);

  LibkinetoApi& libkinetoApi_;
  std::mutex configLock_;
  const char* configFileName_;
  Config config_;
  std::unique_ptr<Config> onDemandEventProfilerConfig_;
  std::unique_ptr<DaemonConfigLoader> daemonConfigLoader_;

  std::chrono::seconds configUpdateIntervalSecs_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;
  std::unique_ptr<std::thread> updateThread_;
  std::condition_variable updateThreadCondVar_;
  std::mutex updateThreadMutex_;
  std::atomic_bool stopFlag_{false};
  std::atomic_bool onDemandSignal_{false};
};

} // namespace KINETO_NAMESPACE
