/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <mutex>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityProfilerProxy.h"
#include "Config.h"
#include "DaemonConfigLoader.h"
#ifdef HAS_CUPTI
#include "CuptiCallbackApi.h"
#include "CuptiActivityApi.h"
#include "CuptiRangeProfiler.h"
#include "EventProfilerController.h"
#endif
#include "cupti_call.h"
#include "libkineto.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

#ifdef HAS_CUPTI
static bool initialized = false;
static std::mutex initMutex;

bool enableEventProfiler() {
  if (getenv("KINETO_ENABLE_EVENT_PROFILER") != nullptr) {
    return true;
  } else {
    return false;
  }
}

static void initProfilers(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* cbInfo) {
  VLOG(0) << "CUDA Context created";
  std::lock_guard<std::mutex> lock(initMutex);

  if (!initialized) {
    libkineto::api().initProfilerIfRegistered();
    initialized = true;
    VLOG(0) << "libkineto profilers activated";
  }

  if (!enableEventProfiler()) {
    VLOG(0) << "Kineto EventProfiler disabled, skipping start";
    return;
  } else {
    CUpti_ResourceData* d = (CUpti_ResourceData*)cbInfo;
    CUcontext ctx = d->context;
    ConfigLoader& config_loader = libkineto::api().configLoader();
    config_loader.initBaseConfig();
    auto config = config_loader.getConfigCopy();
    if (config->eventProfilerEnabled()) {
      EventProfilerController::start(ctx, config_loader);
      LOG(INFO) << "Kineto EventProfiler started";
    }
  }
}

// Some models suffer from excessive instrumentation code gen
// on dynamic attach which can hang for more than 5+ seconds.
// If the workload was meant to be traced, preload the CUPTI
// to take the performance hit early on.
// https://docs.nvidia.com/cupti/r_main.html#r_overhead
static bool shouldPreloadCuptiInstrumentation() {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11020
  return true;
#else
  return false;
#endif
}

static void stopProfiler(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* cbInfo) {
  VLOG(0) << "CUDA Context destroyed";
  std::lock_guard<std::mutex> lock(initMutex);

  if (enableEventProfiler()) {
    CUpti_ResourceData* d = (CUpti_ResourceData*)cbInfo;
    CUcontext ctx = d->context;
    EventProfilerController::stopIfEnabled(ctx);
    LOG(INFO) << "Kineto EventProfiler stopped";
  }
}

static std::unique_ptr<CuptiRangeProfilerInit> rangeProfilerInit;
#endif // HAS_CUPTI

} // namespace KINETO_NAMESPACE

// Callback interface with CUPTI and library constructors
using namespace KINETO_NAMESPACE;
extern "C" {

// Return true if no CUPTI errors occurred during init
void libkineto_init(bool cpuOnly, bool logOnError) {
  // Start with initializing the log level
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  if (logLevelEnv) {
    // atoi returns 0 on error, so that's what we want - default to VERBOSE
    static_assert (static_cast<int>(VERBOSE) == 0, "");
    SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
  }

  // Factory to connect to open source daemon if present
#if __linux__
  if (getenv(kUseDaemonEnvVar) != nullptr) {
    LOG(INFO) << "Registering daemon config loader";
    DaemonConfigLoader::registerFactory();
  }
#endif

#ifdef HAS_CUPTI
  if (!cpuOnly) {
    // libcupti will be lazily loaded on this call.
    // If it is not available (e.g. CUDA is not installed),
    // then this call will return an error and we just abort init.
    auto cbapi = CuptiCallbackApi::singleton();
    cbapi->initCallbackApi();
    bool status = false;
    bool initRangeProfiler = true;

    if (cbapi->initSuccess()){
      const CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RESOURCE;
      status = cbapi->registerCallback(
          domain, CuptiCallbackApi::RESOURCE_CONTEXT_CREATED, initProfilers);
      status = status && cbapi->registerCallback(
          domain, CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED, stopProfiler);

      if (status) {
        status = cbapi->enableCallback(
            domain, CuptiCallbackApi::RESOURCE_CONTEXT_CREATED);
        status = status && cbapi->enableCallback(
            domain, CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED);
      }
    }

    if (!cbapi->initSuccess() || !status) {
      initRangeProfiler = false;
      cpuOnly = true;
      if (logOnError) {
        CUPTI_CALL(cbapi->getCuptiStatus());
        LOG(WARNING) << "CUPTI initialization failed - "
                     << "CUDA profiler activities will be missing";
        LOG(INFO) << "If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to "
                  << "https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti";
      }
    }

    // initialize CUPTI Range Profiler API
    if (initRangeProfiler) {
      rangeProfilerInit = std::make_unique<CuptiRangeProfilerInit>();
    }
  }

  if (shouldPreloadCuptiInstrumentation()) {
    CuptiActivityApi::forceLoadCupti();
  }
#endif // HAS_CUPTI

  ConfigLoader& config_loader = libkineto::api().configLoader();
  libkineto::api().registerProfiler(
      std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));

}

// The cuda driver calls this function if the CUDA_INJECTION64_PATH environment
// variable is set
int InitializeInjection(void) {
  LOG(INFO) << "Injection mode: Initializing libkineto";
  libkineto_init(false /*cpuOnly*/, true /*logOnError*/);
  return 1;
}

void suppressLibkinetoLogMessages() {
  // Only suppress messages if explicit override wasn't provided
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  if (!logLevelEnv || !*logLevelEnv) {
    SET_LOG_SEVERITY_LEVEL(ERROR);
  }
}

} // extern C
