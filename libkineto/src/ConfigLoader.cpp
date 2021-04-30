/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ConfigLoader.h"

#ifdef __linux__
#include <signal.h>
#endif

#include <stdlib.h>
#include <chrono>
#include <fstream>

#include "libkineto.h"
#include "ActivityProfilerProxy.h"
#include "DaemonConfigLoader.h"

#include "Logger.h"

using namespace std::chrono;
using std::string;

namespace KINETO_NAMESPACE {

using namespace libkineto;

const string kConfigFileEnvVar = "KINETO_CONFIG";
#ifdef __linux__
const string kConfigFile = "/etc/libkineto.conf";
const string kOnDemandConfigFile = "/tmp/libkineto.conf";
#else
const string kConfigFile = "libkineto.conf";
const string kOnDemandConfigFile = "libkineto.conf";
#endif

constexpr std::chrono::seconds kConfigUpdateIntervalSecs(300);
constexpr std::chrono::seconds kOnDemandConfigUpdateIntervalSecs(5);
constexpr std::chrono::seconds kOnDemandConfigVerboseLogDurationSecs(120);

#ifdef __linux__
static struct sigaction originalUsr2Handler = {};
#endif

// Use SIGUSR2 to initiate profiling.
// Look for an on-demand config file.
// If none is found, default to base config.
// Try to not affect existing handlers
static bool hasOriginalSignalHandler() {
#ifdef __linux__
  return originalUsr2Handler.sa_handler != nullptr ||
      originalUsr2Handler.sa_sigaction != nullptr;
#else
  return false;
#endif
}

static void handle_signal(int signal) {
#ifdef __linux__
  if (signal == SIGUSR2) {
    ConfigLoader::instance().handleOnDemandSignal();
    if (hasOriginalSignalHandler()) {
      // Invoke original handler and reinstate ours
      struct sigaction act;
      sigaction(SIGUSR2, &originalUsr2Handler, &act);
      raise(SIGUSR2);
      sigaction(SIGUSR2, &act, &originalUsr2Handler);
    }
  }
#endif
}

static void setupSignalHandler(bool enableSigUsr2) {
#ifdef __linux__
  if (enableSigUsr2) {
    struct sigaction act = {};
    act.sa_handler = &handle_signal;
    act.sa_flags = SA_NODEFER;
    if (sigaction(SIGUSR2, &act, &originalUsr2Handler) < 0) {
      PLOG(ERROR) << "Failed to register SIGUSR2 handler";
    }
    if (originalUsr2Handler.sa_handler == &handle_signal) {
      originalUsr2Handler = {};
    }
  } else if (hasOriginalSignalHandler()) {
    sigaction(SIGUSR2, &originalUsr2Handler, nullptr);
    originalUsr2Handler = {};
  }
#endif
}

// return an empty string if reading gets any errors. Otherwise a config string.
static std::string readConfigFromConfigFile(const char* filename) {
  // Read whole file into a string.
  std::ifstream file(filename);
  std::string conf;
  try {
    conf.assign(
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } catch (std::exception& e) {
    LOG(ERROR) << "Error in reading libkineto config from config file: "
               << e.what();
    conf = "";
  }
  return conf;
}

static std::function<std::unique_ptr<DaemonConfigLoader>()>&
daemonConfigLoaderFactory() {
  static std::function<std::unique_ptr<DaemonConfigLoader>()> factory = nullptr;
  return factory;
}

void ConfigLoader::setDaemonConfigLoaderFactory(
    std::function<std::unique_ptr<DaemonConfigLoader>()> factory) {
  daemonConfigLoaderFactory() = factory;
}

ConfigLoader& ConfigLoader::instance() {
  static ConfigLoader config_loader(libkineto::api());
  return config_loader;
}

// return an empty string if polling gets any errors. Otherwise a config string.
std::string ConfigLoader::readOnDemandConfigFromDaemon(
    time_point<system_clock> now) {
  if (!daemonConfigLoader_) {
    return "";
  }
  bool events =
      now > onDemandEventProfilerConfig_->eventProfilerOnDemandEndTime();
  bool activities = !libkinetoApi_.activityProfiler().isActive();
  return daemonConfigLoader_->readOnDemandConfig(events, activities);
}

int ConfigLoader::contextCountForGpu(uint32_t device) {
  if (!daemonConfigLoader_) {
    // FIXME: Throw error?
    return 0;
  }
  return daemonConfigLoader_->gpuContextCount(device);
}

ConfigLoader::ConfigLoader(LibkinetoApi& api)
    : libkinetoApi_(api),
      onDemandEventProfilerConfig_(new Config()),
      configUpdateIntervalSecs_(kConfigUpdateIntervalSecs),
      onDemandConfigUpdateIntervalSecs_(kOnDemandConfigUpdateIntervalSecs),
      stopFlag_(false),
      onDemandSignal_(false) {
  configFileName_ = getenv(kConfigFileEnvVar.data());
  if (configFileName_ == nullptr) {
    configFileName_ = kConfigFile.data();
  }
  config_.parse(readConfigFromConfigFile(configFileName_));
  SET_LOG_VERBOSITY_LEVEL(config_.verboseLogLevel(), config_.verboseLogModules());
  setupSignalHandler(config_.sigUsr2Enabled());
  if (daemonConfigLoaderFactory && daemonConfigLoaderFactory()) {
    daemonConfigLoader_ = daemonConfigLoaderFactory()();
    daemonConfigLoader_->setCommunicationFabric(config_.ipcFabricEnabled());
  }
  updateThread_ =
      std::make_unique<std::thread>(&ConfigLoader::updateConfigThread, this);
}

ConfigLoader::~ConfigLoader() {
  if (updateThread_) {
    stopFlag_ = true;
    {
      std::lock_guard<std::mutex> lock(updateThreadMutex_);
      updateThreadCondVar_.notify_one();
    }
    updateThread_->join();
  }
}

void ConfigLoader::handleOnDemandSignal() {
  onDemandSignal_ = true;
  {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    updateThreadCondVar_.notify_one();
  }
}

void ConfigLoader::updateBaseConfig() {
  const std::string config_str = readConfigFromConfigFile(configFileName_);
  if (config_str != config_.source()) {
    std::lock_guard<std::mutex> lock(configLock_);
    config_.~Config();
    new (&config_) Config();
    config_.parse(config_str);
    if (daemonConfigLoader_) {
      daemonConfigLoader_->setCommunicationFabric(config_.ipcFabricEnabled());
    }
  }
  setupSignalHandler(config_.sigUsr2Enabled());
}

void ConfigLoader::configureFromSignal(
    time_point<system_clock> now,
    Config& config) {
  LOG(INFO) << "Received on-demand profiling signal, "
            << "reading config from " << kOnDemandConfigFile.data();
  const std::string config_str =
      readConfigFromConfigFile(kOnDemandConfigFile.data());
  config.parse(config_str);
  config.setSignalDefaults();
  if (daemonConfigLoader_) {
    daemonConfigLoader_->setCommunicationFabric(config_.ipcFabricEnabled());
  }
  if (eventProfilerRequest(config)) {
    if (now > onDemandEventProfilerConfig_->eventProfilerOnDemandEndTime()) {
      LOG(INFO) << "Starting on-demand event profiling from signal";
      std::lock_guard<std::mutex> lock(configLock_);
      onDemandEventProfilerConfig_ = config.clone();
    } else {
      LOG(ERROR) << "On-demand event profiler is busy";
    }
  }
  // Initiate a trace by default, even when not specified in the config.
  // Set trace duration and iterations to 0 to suppress.
  config.updateActivityProfilerRequestReceivedTime();
  try {
    auto& profiler = dynamic_cast<ActivityProfilerProxy&>(
        libkinetoApi_.activityProfiler());
    profiler.scheduleTrace(config);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to schedule profiler request (busy?)";
  }
}

void ConfigLoader::configureFromDaemon(
    time_point<system_clock> now,
    Config& config) {
  const std::string config_str = readOnDemandConfigFromDaemon(now);
  LOG_IF(INFO, !config_str.empty()) << "Received config from dyno:\n"
                                    << config_str;
  config.parse(config_str);
  if (daemonConfigLoader_) {
    daemonConfigLoader_->setCommunicationFabric(config_.ipcFabricEnabled());
  }
  if (eventProfilerRequest(config)) {
    std::lock_guard<std::mutex> lock(configLock_);
    onDemandEventProfilerConfig_ = config.clone();
  }
  if (config_.activityProfilerEnabled() &&
      config.activityProfilerRequestReceivedTime() > now) {
    try {
      auto& profiler = dynamic_cast<ActivityProfilerProxy&>(
          libkinetoApi_.activityProfiler());
      profiler.scheduleTrace(config);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to schedule profiler request (busy?)";
    }
  }
}

void ConfigLoader::updateConfigThread() {
  auto now = system_clock::now();
  auto next_config_load_time = now + configUpdateIntervalSecs_;
  auto next_on_demand_load_time = now + onDemandConfigUpdateIntervalSecs_;
  auto next_log_level_reset_time = now;
  seconds interval = configUpdateIntervalSecs_;
  if (interval > onDemandConfigUpdateIntervalSecs_) {
    interval = onDemandConfigUpdateIntervalSecs_;
  }
  auto onDemandConfig = std::make_unique<Config>();

  // This can potentially sleep for long periods of time, so allow
  // the desctructor to wake it to avoid a 5-minute long destruct period.
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(updateThreadMutex_);
      updateThreadCondVar_.wait_for(lock, interval);
    }
    if (stopFlag_) {
      break;
    }
    now = system_clock::now();
    if (now > next_config_load_time) {
      updateBaseConfig();
      next_config_load_time = now + configUpdateIntervalSecs_;
    }
    if (onDemandSignal_.exchange(false)) {
      onDemandConfig = config_.clone();
      configureFromSignal(now, *onDemandConfig);
    } else if (now > next_on_demand_load_time) {
      configureFromDaemon(now, *onDemandConfig);
      next_on_demand_load_time = now + onDemandConfigUpdateIntervalSecs_;
    }
    if (onDemandConfig->verboseLogLevel() >= 0) {
      LOG(INFO) << "Setting verbose level to "
                << onDemandConfig->verboseLogLevel()
                << " from on-demand config";
      SET_LOG_VERBOSITY_LEVEL(
          onDemandConfig->verboseLogLevel(),
          onDemandConfig->verboseLogModules());
      next_log_level_reset_time = now + kOnDemandConfigVerboseLogDurationSecs;
    }
    if (now > next_log_level_reset_time) {
      VLOG(0) << "Resetting verbose level";
      SET_LOG_VERBOSITY_LEVEL(
          config_.verboseLogLevel(), config_.verboseLogModules());
    }
  }
}

bool ConfigLoader::hasNewConfig(const Config& oldConfig) {
  std::lock_guard<std::mutex> lock(configLock_);
  return config_.timestamp() > oldConfig.timestamp();
}

bool ConfigLoader::hasNewEventProfilerOnDemandConfig(const Config& oldConfig) {
  std::lock_guard<std::mutex> lock(configLock_);
  return onDemandEventProfilerConfig_->eventProfilerOnDemandStartTime() >
      oldConfig.eventProfilerOnDemandStartTime();
}

} // namespace KINETO_NAMESPACE
