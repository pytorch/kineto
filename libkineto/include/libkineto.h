// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
#include <deque>

#include "ActivityProfilerInterface.h"
#include "ActivityType.h"
#include "ClientInterface.h"
#include "GenericTraceActivity.h"
#include "TraceSpan.h"
#include "IActivityProfiler.h"
#include "ActivityTraceInterface.h"

#include "ThreadUtil.h"

extern "C" {
  void suppressLibkinetoLogMessages();
  int InitializeInjection(void);
  bool libkineto_init(bool cpuOnly, bool logOnError);
}

namespace libkineto {

class Config;
class ConfigLoader;

struct CpuTraceBuffer {
  TraceSpan span{0, 0, "none"};
  int gpuOpCount;
  std::deque<GenericTraceActivity> activities;
};

using ChildActivityProfilerFactory =
  std::function<std::unique_ptr<IActivityProfiler>()>;

class LibkinetoApi {
 public:

  explicit LibkinetoApi(ConfigLoader& configLoader)
      : configLoader_(configLoader) {
  }

  // Called by client that supports tracing API.
  // libkineto can still function without this.
  void registerClient(ClientInterface* client);

  // Called by libkineto on init
  void registerProfiler(std::unique_ptr<ActivityProfilerInterface> profiler) {
    activityProfiler_ = std::move(profiler);
    initClientIfRegistered();
  }

  ActivityProfilerInterface& activityProfiler() {
    return *activityProfiler_;
  }

  ClientInterface* client() {
    return client_;
  }

  void initProfilerIfRegistered() {
    static std::once_flag once;
    if (activityProfiler_) {
      std::call_once(once, [this] {
        if (!activityProfiler_->isInitialized()) {
          activityProfiler_->init();
          initChildActivityProfilers();
        }
      });
    }
  }

  bool isProfilerInitialized() const {
    return activityProfiler_ && activityProfiler_->isInitialized();
  }

  bool isProfilerRegistered() const {
    return activityProfiler_ != nullptr;
  }

  void suppressLogMessages() {
    suppressLibkinetoLogMessages();
  }

  // Provides access to profier configuration manaegement
  ConfigLoader& configLoader() {
    return configLoader_;
  }

  void registerProfilerFactory(
      ChildActivityProfilerFactory factory) {
    if (isProfilerInitialized()) {
      activityProfiler_->addChildActivityProfiler(factory());
    } else {
      childProfilerFactories_.push_back(factory);
    }
  }

 private:

  void initChildActivityProfilers() {
    if (!isProfilerInitialized()) {
      return;
    }
    for (const auto& factory : childProfilerFactories_) {
      activityProfiler_->addChildActivityProfiler(factory());
    }
    childProfilerFactories_.clear();
  }

  // Client is initialized once both it and libkineto has registered
  void initClientIfRegistered();

  ConfigLoader& configLoader_;
  std::unique_ptr<ActivityProfilerInterface> activityProfiler_{};
  ClientInterface* client_{};
  int32_t clientRegisterThread_{0};

  bool isLoaded_{false};
  std::vector<ChildActivityProfilerFactory> childProfilerFactories_;
};

// Singleton
LibkinetoApi& api();

} // namespace libkineto
