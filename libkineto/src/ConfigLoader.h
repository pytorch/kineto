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

namespace libkineto {
class LibkinetoApi;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
class IDaemonConfigLoader;

class ConfigLoader {
 public:
  static ConfigLoader& instance();

  enum ConfigKind { ActivityProfiler = 0, EventProfiler, NumConfigKinds };

  struct ConfigHandler {
    virtual ~ConfigHandler() = default;
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
    auto it = std::find(handlers_[kind].begin(), handlers_[kind].end(), handler);
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

  std::unique_ptr<Config> getConfigCopy() {
    std::lock_guard<std::mutex> lock(configLock_);
    return config_->clone();
  }

  bool hasNewConfig(const Config& oldConfig);
  int contextCountForGpu(uint32_t device);

  void handleOnDemandSignal();

  // Add a config loader factory. Multiple loaders can coexist (e.g.,
  // DaemonConfigLoader for IPC-based Dynolog + PortConfigLoader for TCP).
  // Each factory will be invoked once to create a loader instance.
  static void addConfigLoaderFactory(std::function<std::unique_ptr<IDaemonConfigLoader>()> factory);

  std::string getConfString();

  // ============================================================================
  // Test-only APIs
  // ============================================================================
  // These methods exist solely to enable unit testing of the multi-loader
  // infrastructure. The ConfigLoader is a singleton with static factory
  // storage, which makes isolated testing impossible without reset
  // capabilities.
  //
  // Why these are needed:
  // 1. configLoaderFactories() is a static vector that persists across tests.
  //    Without clearConfigLoaderFactories(), factories registered in one test
  //    would leak into subsequent tests, causing non-deterministic behavior.
  //
  // 2. configLoaders_ is populated lazily via initConfigLoaders(). Without
  //    clearConfigLoaders(), loaders created in one test would persist,
  //    preventing tests from verifying fresh loader creation.
  //
  // 3. These APIs allow testing:
  //    - Multiple factories registered → multiple loaders created
  //    - First successful loader's config is returned
  //    - Empty config when no loader has data
  //
  // Production code should NEVER call these methods.
  // ============================================================================
  static void clearConfigLoaderFactories();
  void clearConfigLoaders();

 private:
  ConfigLoader();
  ~ConfigLoader();

  // Initialize all config loaders from registered factories
  void initConfigLoaders();

  void startThread();
  void stopThread();
  void updateConfigThread();
  void updateBaseConfig();

  // Create configuration when receiving SIGUSR2
  void configureFromSignal(std::chrono::time_point<std::chrono::system_clock> now, Config& config);

  // Create configuration when receiving request from a daemon or port-based
  // loader
  void configureFromDaemon(std::chrono::time_point<std::chrono::system_clock> now, Config& config);

  std::string readOnDemandConfigFromDaemon(std::chrono::time_point<std::chrono::system_clock> now);

  const char* customConfigFileName();

  std::mutex configLock_;
  std::unique_ptr<Config> config_;

  // Support multiple config loaders (e.g., DaemonConfigLoader +
  // PortConfigLoader)
  std::vector<std::unique_ptr<IDaemonConfigLoader>> configLoaders_;

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
