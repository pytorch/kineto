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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "Config.h"

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace libkineto {
  class LibkinetoApi;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
class IDaemonConfigLoader;

class ConfigLoader {
 public:

  static ConfigLoader& instance();

  enum ConfigKind {
    ActivityProfiler = 0,
    EventProfiler,
    NumConfigKinds
  };

  struct ConfigHandler {
    virtual ~ConfigHandler() {}
    virtual bool canAcceptConfig() = 0;
    virtual void acceptConfig(const Config& cfg) = 0;
  };

  void addHandler(ConfigKind kind, ConfigHandler* handler) {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    handlers_[kind].push_back(handler);
    startThread();
  }

  void removeHandler(ConfigKind kind, ConfigHandler* handler) {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    auto it = std::find(
        handlers_[kind].begin(), handlers_[kind].end(), handler);
    if (it != handlers_[kind].end()) {
      handlers_[kind].erase(it);
    }
  }

  void notifyHandlers(const Config& cfg) {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    for (auto& key_val : handlers_) {
      for (ConfigHandler* handler : key_val.second) {
        handler->acceptConfig(cfg);
      }
    }
  }

  bool canHandlerAcceptConfig(ConfigKind kind) {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    for (ConfigHandler* handler : handlers_[kind]) {
      if (!handler->canAcceptConfig()) {
        return false;
      }
    }
    return true;
  }

  void initBaseConfig() {
    bool init = false;
    {
      std::lock_guard<std::mutex> lock(configLock_);
      init = !config_ || config_->source().empty();
    }
    if (init) {
      updateBaseConfig();
    }
  }

  inline std::unique_ptr<Config> getConfigCopy() {
    std::lock_guard<std::mutex> lock(configLock_);
    return config_->clone();
  }

  bool hasNewConfig(const Config& oldConfig);
  int contextCountForGpu(uint32_t gpu);

  void handleOnDemandSignal();

  static void setDaemonConfigLoaderFactory(
      std::function<std::unique_ptr<IDaemonConfigLoader>()> factory);

  const std::string getConfString();

 private:
  ConfigLoader();
  ~ConfigLoader();

  IDaemonConfigLoader* daemonConfigLoader();

  void startThread();
  void stopThread();
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

  std::string readOnDemandConfigFromDaemon(
      std::chrono::time_point<std::chrono::system_clock> now);

  const char* customConfigFileName();

  std::mutex configLock_;
  std::unique_ptr<Config> config_;
  std::unique_ptr<IDaemonConfigLoader> daemonConfigLoader_;
  std::map<ConfigKind, std::vector<ConfigHandler*>> handlers_;

  std::chrono::seconds configUpdateIntervalSecs_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;
  std::unique_ptr<std::thread> updateThread_;
  std::condition_variable updateThreadCondVar_;
  std::mutex updateThreadMutex_;
  std::atomic_bool stopFlag_{false};
  std::atomic_bool onDemandSignal_{false};
};

} // namespace KINETO_NAMESPACE
