/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "external_api.h"

namespace libkineto {

external_api& external_api::getInstance() {
  static external_api gInstance;
  return gInstance;
}

void external_api::initTracer() {
  if (getInstance().externalInit_) {
    if (getInstance().externalInitThread_ != pthread_self()) {
      fprintf(
          stderr,
          "ERROR: External init callback must run in same thread as registerTracer "
          "(%d != %d)\n",
          (int)pthread_self(),
          (int)getInstance().externalInitThread_);
    } else {
      getInstance().externalInit_();
      getInstance().externalInit_ = nullptr;
    }
  }
}

void external_api::initialize(
    std::function<void(std::unique_ptr<CpuTraceBuffer>)> transferFunc,
    std::function<void(int)> pushCorrelationIDFunc,
    std::function<void(void)> popCorrelationIDFunc,
    std::function<bool(const std::string&)> netNameFilterFunc) {
  getInstance().transferCpuTrace_ = transferFunc;
  getInstance().pushCorrelationID_ = pushCorrelationIDFunc;
  getInstance().popCorrelationID_ = popCorrelationIDFunc;
  getInstance().netNameFilter_ = netNameFilterFunc;

  initTracer();
}

// Has libkineto loaded yet?
// Called from application before call to registerTracer.
// This could return false from a static function that gets executed
// before libkineto has a chance to load. In that case, pass a lazy
// init callback to registerTracer.
bool external_api::isLoaded() {
  return getInstance().isLoaded_;
}

void external_api::setLoaded(std::function<void(void)> initFunc) {
  // Function passed from libkineto to lazy init libkineto
  getInstance().libkinetoInit_ = initFunc;
  getInstance().isLoaded_ = true;
  // If tracer registered before this, init it now
  initTracer();
}

// Called by libkineto to check whether an external profiler
// has registered itself
bool external_api::isSupported() {
  // Check some state set in registerTracer
  return getInstance().externalInitThread_ != 0;
}

void external_api::registerTracer(std::function<void(void)> initFunc) {
  if (initFunc && getInstance().isLoaded()) {
    // Can initialize straight away
    initFunc();
  } else {
    // Delay init until libkineto actually starts, we don't want to
    // activate the external tracer when it definitely won't be used.
    getInstance().externalInit_ = initFunc;
  }
  // Assume here that the external init callback is *not* threadsafe
  // and only call it if it's the same thread that called registerTracer
  getInstance().externalInitThread_ = pthread_self();
}

bool external_api::enableForNet(const std::string& name) {
  return getInstance().netNameFilter_(name);
}

void external_api::setProfileRequestActive(bool active) {
  getInstance().requestActive_ = active;
}

int external_api::netSizeThreshold() {
  return getInstance().netSizeThreshold_;
}

void external_api::setNetSizeThreshold(int ops) {
  getInstance().netSizeThreshold_ = ops;
}

void external_api::transferCpuTrace(
    std::unique_ptr<external_api::CpuTraceBuffer> cpuTrace) {
  getInstance().transferCpuTrace_(std::move(cpuTrace));
}

void external_api::pushCorrelationID(int id) {
  getInstance().pushCorrelationID_(id);
}

void external_api::popCorrelationID() {
  getInstance().popCorrelationID_();
}

} // namespace libkineto
