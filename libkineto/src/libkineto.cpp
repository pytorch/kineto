/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * This library performs basic cupti event collection and reporting.
 *
 * Usage:
 * Library can be built as a standalone shared library or for inclusion in a
 * cuda binary using the libkineto.so and kineto build targets respectively.
 *
 * When included in a cuda binary, the library is initialized upon loading
 * by dlopen().
 * When used as a standalone library, it can be loaded by setting the
 * CUDA_INJECTION64_PATH environment variable (for the target process) to point
 * at the library, and the cuda driver will load it.
 *
 * Which events to profile can be specified in the config file pointed to
 * by KINETO_CONFIG as a comma separated list. See cupti documentation for
 * event names.
 *
 * The library will fail to initialize when no GPU is present on the system
 * (most likely because libcupti.so will not be found by the lazy loading
 * mechanism), but allows the application to continue.
 */

#include <unistd.h>
#include <iostream>
#include <memory>
#include <mutex>

#include <cupti.h>

#include "ActivityProfilerController.h"
#include "Config.h"
#include "ConfigLoader.h"
#include "EventProfilerController.h"
#include "cupti_call.h"
#include "external_api.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

static bool initialized = false;
static std::mutex initMutex;
static bool loadedByCuda = false;
static void initProfilers(CUcontext ctx) {
  std::lock_guard<std::mutex> lock(initMutex);
  if (!initialized) {
    ActivityProfilerController::init(/* cpuOnly */ false);
    initialized = true;
    VLOG(0) << "libkineto profilers activated";
  }
  EventProfilerController::start(ctx);
}

static void stopProfiler(CUcontext ctx) {
  std::lock_guard<std::mutex> lock(initMutex);
  EventProfilerController::stop(ctx);
}

bool hasConfigEnvVar() {
  return getenv("KINETO_CONFIG") != nullptr;
}

bool hasKnownJobIdEnvVar() {
  // FIXME: Find better way to auto-enable
  // E.g. add FEATURE_AUTO_INIT
  return getenv("CHRONOS_JOB_INSTANCE_ID");
}

} // namespace KINETO_NAMESPACE

// Callback interface with CUPTI and library constructors
using namespace KINETO_NAMESPACE;
extern "C" {

static void CUPTIAPI callback(
    void* /* unused */,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  VLOG(0) << "Callback: domain = " << domain << ", cbid = " << cbid;

  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    CUpti_ResourceData* d = (CUpti_ResourceData*)cbInfo;
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      VLOG(0) << "CUDA Context created";
      initProfilers(d->context);
    } else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      VLOG(0) << "CUDA Context destroyed";
      stopProfiler(d->context);
    }
  }
}

static void libkineto_init(void) {
  // Can be more verbose when injected dynamically
  LOG_IF(INFO, loadedByCuda) << "Initializing libkineto ";
  bool enable = hasConfigEnvVar() || hasKnownJobIdEnvVar();
  CUpti_SubscriberHandle subscriber;
  CUptiResult status = CUPTI_ERROR_UNKNOWN;
  // libcupti will be lazily loaded on this call.
  // If it is not available (e.g. CUDA is not installed),
  // then this call will return an error and we just abort init.
  status = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback, nullptr);
  if (status == CUPTI_SUCCESS) {
    status = cuptiEnableCallback(
        1,
        subscriber,
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
    if (loadedByCuda) {
      CUPTI_CALL(status);
    }
    status = cuptiEnableCallback(
        1,
        subscriber,
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);
    if (loadedByCuda) {
      CUPTI_CALL(status);
    }
  } else if (!enable) {
    // Not explicitly enabled and no GPU present - do not enable external API
    return;
  }

  // notify the external_api libkineto is loaded, so that we'll attach
  // observer to the network.
  // Pass in activity profiler initializer, used to lazily start
  // the activity profiler and config reader threads.
  // It's risky to start them here, since they may use glog and other
  // functionality that needs explicit initialization.
  libkineto::external_api::setLoaded([status] {
    auto config = ConfigLoader::instance().getConfigCopy();
    if (config->activityProfilerEnabled()) {
      bool cpu_only = (status != CUPTI_SUCCESS);
      ActivityProfilerController::init(cpu_only);
    }
  });
}

#define CONCAT(a, b) a##b
#define FUNCNAME(a, b) CONCAT(a, b)
#define LIBKINETO_CONSTRUCTOR FUNCNAME(KINETO_NAMESPACE, _create)
#define LIBKINETO_DESTRUCTOR FUNCNAME(KINETO_NAMESPACE, _destroy)

// dlopen() will call this function before returning
__attribute__((constructor)) void libkineto_create(void) {
  // If CUDA_INJECTION64_PATH is set, don't initialize the library
  // since we're about to load a dynamic version
  if (getenv("CUDA_INJECTION64_PATH") == nullptr) {
    libkineto_init();
  }
}

// dlclose() will call this function before returning
// It's also called when the program exits
__attribute__((destructor)) void libkineto_destroy(void) {
  LOG_IF(INFO, loadedByCuda) << "Destroying libkineto";
}

// The cuda driver calls this function if the CUDA_INJECTION64_PATH environment
// variable is set
int InitializeInjection(void) {
  loadedByCuda = true;
  libkineto_init();
  return 1;
}

} // extern C
