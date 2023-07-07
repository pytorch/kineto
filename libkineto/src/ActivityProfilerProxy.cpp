/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerProxy.h"

#include "ActivityProfilerController.h"
#include "Config.h"
#include "CuptiActivityApi.h"
#include "Logger.h"
#include <chrono>
#ifdef HAS_ROCTRACER
#include "RoctracerActivityApi.h"
#endif

namespace KINETO_NAMESPACE {

ActivityProfilerProxy::ActivityProfilerProxy(
    bool cpuOnly, ConfigLoader& configLoader)
  : cpuOnly_(cpuOnly), configLoader_(configLoader) {
}

ActivityProfilerProxy::~ActivityProfilerProxy() {
  delete controller_;
};

void ActivityProfilerProxy::init() {
  if (!controller_) {
    controller_ = new ActivityProfilerController(configLoader_, cpuOnly_);
  }
}

void ActivityProfilerProxy::scheduleTrace(const std::string& configStr) {
  Config config;
  config.parse(configStr);
  controller_->scheduleTrace(config);
}

void ActivityProfilerProxy::scheduleTrace(const Config& config) {
  controller_->scheduleTrace(config);
}

void ActivityProfilerProxy::prepareTrace(
    const std::set<ActivityType>& activityTypes,
    const std::string& configStr) {
  Config config;
  bool validate_required = true;

  // allow user provided config (ExperimentalConfig)to override default options
  if (!configStr.empty()) {
    if (!config.parse(configStr)) {
      LOG(WARNING) << "Failed to parse config : " << configStr;
    }
    // parse also runs validate
    validate_required = false;
  }
  // user provided config (KINETO_CONFIG) to override default options
  auto loaded_config = configLoader_.getConfString();
  if (!loaded_config.empty()) {
    config.parse(loaded_config);
  }

  config.setClientDefaults();
  config.setSelectedActivityTypes(activityTypes);

  if (validate_required) {
    config.validate(std::chrono::system_clock::now());
  }

  controller_->prepareTrace(config);
}

void ActivityProfilerProxy::startTrace() {
  controller_->startTrace();
}

std::unique_ptr<ActivityTraceInterface>
ActivityProfilerProxy::stopTrace() {
  return controller_->stopTrace();
}

void ActivityProfilerProxy::step() {
  controller_->step();
}

bool ActivityProfilerProxy::isActive() {
  return controller_->isActive();
}

void ActivityProfilerProxy::pushCorrelationId(uint64_t id) {
  CuptiActivityApi::pushCorrelationID(id,
    CuptiActivityApi::CorrelationFlowType::Default);
#ifdef HAS_ROCTRACER
  // FIXME: bad design here
  RoctracerActivityApi::pushCorrelationID(id,
    RoctracerActivityApi::CorrelationFlowType::Default);
#endif
}

void ActivityProfilerProxy::popCorrelationId() {
  CuptiActivityApi::popCorrelationID(
    CuptiActivityApi::CorrelationFlowType::Default);
#ifdef HAS_ROCTRACER
  RoctracerActivityApi::popCorrelationID(
    RoctracerActivityApi::CorrelationFlowType::Default);
#endif
}

void ActivityProfilerProxy::pushUserCorrelationId(uint64_t id) {
  CuptiActivityApi::pushCorrelationID(id,
    CuptiActivityApi::CorrelationFlowType::User);
}

void ActivityProfilerProxy::popUserCorrelationId() {
  CuptiActivityApi::popCorrelationID(
    CuptiActivityApi::CorrelationFlowType::User);
}

void ActivityProfilerProxy::transferCpuTrace(
   std::unique_ptr<CpuTraceBuffer> traceBuffer) {
  controller_->transferCpuTrace(std::move(traceBuffer));
}

void ActivityProfilerProxy::addMetadata(
    const std::string& key, const std::string& value) {
  controller_->addMetadata(key, value);
}

void ActivityProfilerProxy::recordThreadInfo() {
  controller_->recordThreadInfo();
}

void ActivityProfilerProxy::addChildActivityProfiler(
    std::unique_ptr<IActivityProfiler> profiler) {
  controller_->addChildActivityProfiler(std::move(profiler));
}

void ActivityProfilerProxy::logInvariantViolation(
    const std::string& profile_id,
    const std::string& assertion,
    const std::string& error,
    const std::string& group_profile_id) {
    controller_->logInvariantViolation(profile_id, assertion, error, group_profile_id);
}

} // namespace libkineto
