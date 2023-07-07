/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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
#include <functional>
#include <memory>

#include "DaemonConfigLoader.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {


constexpr char kConfigFileEnvVar[] = "KINETO_CONFIG";
#ifdef __linux__
constexpr char kConfigFile[] = "/etc/libkineto.conf";
constexpr char kOnDemandConfigFile[] = "/tmp/libkineto.conf";
#else
constexpr char kConfigFile[] = "libkineto.conf";
constexpr char kOnDemandConfigFile[] = "libkineto.conf";
#endif

constexpr std::chrono::seconds kConfigUpdateIntervalSecs(300);

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
static std::string readConfigFromConfigFile(const char* filename, bool verbose=true) {
  // Read whole file into a string.
  std::ifstream file(filename);
  std::string conf;
  try {
    conf.assign(
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } catch (std::exception& e) {
    if (verbose) {
      VLOG(0) << "Error reading " << filename << ": " << e.what();
    }

    conf = "";
  }
  return conf;
}

static std::function<std::unique_ptr<IDaemonConfigLoader>()>&
daemonConfigLoaderFactory() {
  static std::function<std::unique_ptr<IDaemonConfigLoader>()> factory = nullptr;
  return factory;
}

void ConfigLoader::setDaemonConfigLoaderFactory(
    std::function<std::unique_ptr<IDaemonConfigLoader>()> factory) {
  daemonConfigLoaderFactory() = factory;
}

ConfigLoader& ConfigLoader::instance() {
  static ConfigLoader config_loader;
  return config_loader;
}

// return an empty string if polling gets any errors. Otherwise a config string.
std::string ConfigLoader::readOnDemandConfigFromDaemon(
    time_point<system_clock> now) {
  if (!daemonConfigLoader_) {
    return "";
  }
  bool events = canHandlerAcceptConfig(ConfigKind::EventProfiler);
  bool activities = canHandlerAcceptConfig(ConfigKind::ActivityProfiler);
  return daemonConfigLoader_->readOnDemandConfig(events, activities);
}

int ConfigLoader::contextCountForGpu(uint32_t device) {
  if (!daemonConfigLoader_) {
    // FIXME: Throw error?
    return 0;
  }
  return daemonConfigLoader_->gpuContextCount(device);
}

ConfigLoader::ConfigLoader()
    : configUpdateIntervalSecs_(kConfigUpdateIntervalSecs),
      // on-demand config will be overwritten by the value read from the regular config
      // so the initial value is not important
      onDemandConfigUpdateIntervalSecs_(kConfigUpdateIntervalSecs),
      stopFlag_(false),
      onDemandSignal_(false) {
}

void ConfigLoader::startThread() {
  if (!updateThread_) {
    // Create default base config here - at this point static initializers
    // of extensions should have run and registered all config feature factories
    std::lock_guard<std::mutex> lock(configLock_);
    if (!config_) {
      config_ = std::make_unique<Config>();
    }
    updateThread_ =
        std::make_unique<std::thread>(&ConfigLoader::updateConfigThread, this);
  }
}

void ConfigLoader::stopThread() {
  if (updateThread_) {
    stopFlag_ = true;
    {
      std::lock_guard<std::mutex> lock(updateThreadMutex_);
      updateThreadCondVar_.notify_one();
    }
    updateThread_->join();
    updateThread_ = nullptr;
  }
}

ConfigLoader::~ConfigLoader() {
  stopThread();
#if !USE_GOOGLE_LOG
  Logger::clearLoggerObservers();
#endif // !USE_GOOGLE_LOG
}

void ConfigLoader::handleOnDemandSignal() {
  onDemandSignal_ = true;
  {
    std::lock_guard<std::mutex> lock(updateThreadMutex_);
    updateThreadCondVar_.notify_one();
  }
}

namespace {

const char* configFileName() {
  static const char* configFileName__ = []() {
    const char* configFileName_ = getenv(kConfigFileEnvVar);
    if (configFileName_ == nullptr) {
      configFileName_ = kConfigFile;
    }
    return configFileName_;
  }();
  return configFileName__;
}

} // namespace

IDaemonConfigLoader* ConfigLoader::daemonConfigLoader() {
  if (!daemonConfigLoader_ && daemonConfigLoaderFactory()) {
    daemonConfigLoader_ = daemonConfigLoaderFactory()();
    daemonConfigLoader_->setCommunicationFabric(config_->ipcFabricEnabled());
  }
  return daemonConfigLoader_.get();
}

const char* ConfigLoader::customConfigFileName() {
  return getenv(kConfigFileEnvVar);
}

const std::string ConfigLoader::getConfString(){
  return readConfigFromConfigFile(configFileName(), false);
}

void ConfigLoader::updateBaseConfig() {
  // First try reading local config file
  // If that fails, read from daemon
  // TODO: Invert these once daemon path fully rolled out
  std::string config_str = readConfigFromConfigFile(configFileName());
  if (config_str.empty() && daemonConfigLoader()) {
    // If local config file was not successfully loaded (e.g. not found)
    // then try the daemon
    config_str = daemonConfigLoader()->readBaseConfig();
  }
  if (config_str != config_->source()) {
    std::lock_guard<std::mutex> lock(configLock_);
    config_ = std::make_unique<Config>();
    config_->parse(config_str);
    if (daemonConfigLoader()) {
      daemonConfigLoader()->setCommunicationFabric(config_->ipcFabricEnabled());
    }
    setupSignalHandler(config_->sigUsr2Enabled());
    SET_LOG_VERBOSITY_LEVEL(
        config_->verboseLogLevel(),
        config_->verboseLogModules());
    VLOG(0) << "Detected base config change";
  }
}

void ConfigLoader::configureFromSignal(
    time_point<system_clock> now,
    Config& config) {
  LOG(INFO) << "Received on-demand profiling signal, "
            << "reading config from " << kOnDemandConfigFile;
  // Reset start time to 0 in order to compute new default start time
  const std::string config_str = "PROFILE_START_TIME=0\n"
      + readConfigFromConfigFile(kOnDemandConfigFile);
  config.parse(config_str);
  config.setSignalDefaults();
  notifyHandlers(config);
}

void ConfigLoader::configureFromDaemon(
    time_point<system_clock> now,
    Config& config) {
  const std::string config_str = readOnDemandConfigFromDaemon(now);
  if (config_str.empty()) {
    return;
  }

  LOG(INFO) << "Received config from dyno:\n" << config_str;
  config.parse(config_str);
  notifyHandlers(config);
}

void ConfigLoader::updateConfigThread() {
  // It's important to hang to this reference until the thread stops.
  // Otherwise, the Config's static members may be destroyed before this
  // function finishes.
  auto handle = Config::getStaticObjectsLifetimeHandle();

  // We're trying to update two configs here:
  // 1. Base config - this is the config that is read from the config file
  // 2. On-demand config - this is the config that is read via IPC channel
  //    from daemon. It's layered on top of the base config.
  // They have different update intervals (on demand is more frequent).
  // Besides, on-demand update frequency can be configured via base config.

  // initialze with some time buffer in the past
  auto prev_config_load_time = system_clock::now() - configUpdateIntervalSecs_ * 2;
  auto prev_on_demand_load_time = prev_config_load_time;
  auto onDemandConfig = std::make_unique<Config>();

  // This can potentially sleep for long periods of time, so allow
  // the destructor to wake it to avoid a 5-minute long destruct period.
  for (;;) {
    auto interval = std::min(
        configUpdateIntervalSecs_ + prev_config_load_time,
        onDemandConfigUpdateIntervalSecs_ + prev_on_demand_load_time) - system_clock::now();
    if (interval.count() > 0) {
      std::unique_lock<std::mutex> lock(updateThreadMutex_);
      updateThreadCondVar_.wait_for(lock, interval);
    }
    if (stopFlag_) {
      break;
    }
    auto now = system_clock::now();
    if (now > prev_config_load_time + configUpdateIntervalSecs_) {
      updateBaseConfig();
      onDemandConfigUpdateIntervalSecs_ = config_->onDemandConfigUpdateIntervalSecs();
      prev_config_load_time = now;
    }
    if (onDemandSignal_.exchange(false)) {
      onDemandConfig = config_->clone();
      configureFromSignal(now, *onDemandConfig);
    } else if (now > prev_on_demand_load_time + onDemandConfigUpdateIntervalSecs_) {
      onDemandConfig = std::make_unique<Config>();
      configureFromDaemon(now, *onDemandConfig);
      prev_on_demand_load_time = now;
    }
    if (onDemandConfig->verboseLogLevel() >= 0) {
      LOG(INFO) << "Setting verbose level to "
                << onDemandConfig->verboseLogLevel()
                << " from on-demand config";
      SET_LOG_VERBOSITY_LEVEL(
          onDemandConfig->verboseLogLevel(),
          onDemandConfig->verboseLogModules());
    }
  }
}

bool ConfigLoader::hasNewConfig(const Config& oldConfig) {
  std::lock_guard<std::mutex> lock(configLock_);
  return config_->timestamp() > oldConfig.timestamp();
}

} // namespace KINETO_NAMESPACE
