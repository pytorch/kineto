// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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
class DaemonConfigLoader;

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
      std::function<std::unique_ptr<DaemonConfigLoader>()> factory);

 private:
  ConfigLoader();
  ~ConfigLoader();

  const char* configFileName();
  DaemonConfigLoader* daemonConfigLoader();

  void startThread();
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

  std::mutex configLock_;
  std::atomic<const char*> configFileName_{nullptr};
  std::unique_ptr<Config> config_;
  std::unique_ptr<DaemonConfigLoader> daemonConfigLoader_;
  std::map<ConfigKind, std::vector<ConfigHandler*>> handlers_;

  std::chrono::seconds configUpdateIntervalSecs_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;
  std::unique_ptr<std::thread> updateThread_;
  std::condition_variable updateThreadCondVar_;
  std::mutex updateThreadMutex_;
  std::atomic_bool stopFlag_{false};
  std::atomic_bool onDemandSignal_{false};

#if !USE_GOOGLE_LOG
  std::unique_ptr<std::set<ILoggerObserver*>> loggerObservers_;
  std::mutex loggerObserversMutex_;
#endif // !USE_GOOGLE_LOG
};

} // namespace KINETO_NAMESPACE
