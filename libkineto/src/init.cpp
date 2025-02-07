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
#include "ConfigLoader.h"
#include "DaemonConfigLoader.h"
#include "DeviceUtil.h"
#include "ThreadUtil.h"
#ifdef HAS_CUPTI
#include "CuptiActivityApi.h"
#include "CuptiCallbackApi.h"
#include "CuptiRangeProfiler.h"
#include "EventProfilerController.h"
#endif
#ifdef HAS_XPUPTI
#include "plugin/xpupti/XpuptiActivityApi.h"
#include "plugin/xpupti/XpuptiActivityProfiler.h"
#endif
#include "libkineto.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

#if __linux__ || defined(HAS_CUPTI)
static bool initialized = false;

static void initProfilers() {
  if (!initialized) {
    libkineto::api().initProfilerIfRegistered();
    libkineto::api().configLoader().initBaseConfig();
    initialized = true;
    VLOG(0) << "libkineto profilers activated";
  }
}

#endif // __linux__ || defined(HAS_CUPTI)

#ifdef HAS_CUPTI
bool enableEventProfiler() {
  if (getenv("KINETO_ENABLE_EVENT_PROFILER") != nullptr) {
    return true;
  } else {
    return false;
  }
}

static void initProfilersCallback(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* /*cbInfo*/) {
  VLOG(0) << "CUDA Context created";
  initProfilers();

  if (enableEventProfiler()) {
    LOG(WARNING) << "Event Profiler is no longer supported in kineto";
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

bool setupCuptiInitCallback(bool logOnError) {
  // libcupti will be lazily loaded on this call.
  // If it is not available (e.g. CUDA is not installed),
  // then this call will return an error and we just abort init.
  auto cbapi = CuptiCallbackApi::singleton();
  cbapi->initCallbackApi();

  bool status = false;

  if (cbapi->initSuccess()) {
    const CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RESOURCE;
    status = cbapi->registerCallback(
        domain,
        CuptiCallbackApi::RESOURCE_CONTEXT_CREATED,
        initProfilersCallback);
    if (status) {
      status = cbapi->enableCallback(
          domain, CuptiCallbackApi::RESOURCE_CONTEXT_CREATED);
    }
  }

  if (!cbapi->initSuccess() || !status) {
    if (logOnError) {
      CUPTI_CALL(cbapi->getCuptiStatus());
      LOG(WARNING) << "CUPTI initialization failed - "
                   << "CUDA profiler activities will be missing";
      LOG(INFO)
          << "If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to "
          << "https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti";
    }
  }

  return status;
}

static std::unique_ptr<CuptiRangeProfilerInit> rangeProfilerInit;
#endif // HAS_CUPTI

} // namespace KINETO_NAMESPACE

// Callback interface with CUPTI and library constructors
using namespace KINETO_NAMESPACE;
extern "C" {

void libkineto_init(bool cpuOnly, bool logOnError) {
  // Start with initializing the log level
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  if (logLevelEnv) {
    // atoi returns 0 on error, so that's what we want - default to VERBOSE
    static_assert(static_cast<int>(VERBOSE) == 0, "");
    SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
  }

  // Factory to connect to open source daemon if present
#if __linux__
  if (libkineto::isDaemonEnvVarSet()) {
    LOG(INFO) << "Registering daemon config loader, cpuOnly =  " << cpuOnly;
    DaemonConfigLoader::registerFactory();
  }
#endif

#ifdef HAS_CUPTI
  bool initRangeProfiler = true;

  if (!cpuOnly && !libkineto::isDaemonEnvVarSet()) {
    bool success = setupCuptiInitCallback(logOnError);
    cpuOnly = !success;
    initRangeProfiler = success;
  }

  // Initialize CUPTI Range Profiler API
  // Note: the following is a no-op if Range Profiler is not supported
  // currently it is only enabled in fbcode.
  if (!cpuOnly && initRangeProfiler) {
    rangeProfilerInit = std::make_unique<CuptiRangeProfilerInit>();
  }

  if (!cpuOnly && shouldPreloadCuptiInstrumentation()) {
    CuptiActivityApi::forceLoadCupti();
  }
#endif // HAS_CUPTI

  ConfigLoader& config_loader = libkineto::api().configLoader();
  libkineto::api().registerProfiler(
      std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));

#ifdef HAS_XPUPTI
  // register xpu pti profiler
  libkineto::api().registerProfilerFactory(
      []() -> std::unique_ptr<IActivityProfiler> {
        auto returnCode = ptiViewGPULocalAvailable();
        if (returnCode != PTI_SUCCESS) {
          std::string errPrefixMsg(
              "Fail to enable Kineto Profiler on XPU due to error code: ");
          errPrefixMsg = errPrefixMsg + std::to_string(returnCode);
#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 9
          std::string errMsg(ptiResultTypeToString(returnCode));
          throw std::runtime_error(
              errPrefixMsg + std::string(". The detailed error message is: ") +
              errMsg);
#else
          throw std::runtime_error(errPrefixMsg);
#endif
        }
        return std::make_unique<XPUActivityProfiler>();
      });
#endif // HAS_XPUPTI

#if __linux__
  // For open source users that would like to connect to a profiling daemon
  // we should always initialize the profiler at this point.
  if (libkineto::isDaemonEnvVarSet()) {
    initProfilers();
  }
#endif
}

// The cuda driver calls this function if the CUDA_INJECTION64_PATH environment
// variable is set. Should be skipped if unset or CUDA_INJECTION64_PATH=none.
int InitializeInjection(void) {
  LOG(INFO) << "Injection mode: Initializing libkineto";
  libkineto_init(false /*cpuOnly*/, true /*logOnError*/);
  return 1;
}

bool hasTestEnvVar() {
  return getenv("GTEST_OUTPUT") != nullptr || getenv("FB_TEST") != nullptr ||
      getenv("PYTORCH_TEST") != nullptr || getenv("TEST_PILOT") != nullptr;
}

void suppressLibkinetoLogMessages() {
  // Only suppress messages if explicit override wasn't provided
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  // For unit tests, don't suppress log verbosity.
  if (!hasTestEnvVar() && (!logLevelEnv || !*logLevelEnv)) {
    SET_LOG_SEVERITY_LEVEL(ERROR);
  }
}

} // extern C
