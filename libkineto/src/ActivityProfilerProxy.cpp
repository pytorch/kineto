/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerProxy.h"

#include "ActivityProfilerController.h"
#include "Config.h"
#include "CuptiActivityInterface.h"

namespace KINETO_NAMESPACE {

void ActivityProfilerProxy::init() {
  if (!controller_) {
    controller_ = new ActivityProfilerController(cpuOnly_);
  }
}

ActivityProfilerProxy::~ActivityProfilerProxy() {
  delete controller_;
};

void ActivityProfilerProxy::scheduleTrace(const std::string& configStr) {
  Config config;
  config.parse(configStr);
  controller_->scheduleTrace(config);
}

void ActivityProfilerProxy::scheduleTrace(const Config& config) {
  controller_->scheduleTrace(config);
}

void ActivityProfilerProxy::prepareTrace(
    const std::set<ActivityType>& activityTypes) {
  Config config;
  config.setClientDefaults();
  config.setSelectedActivityTypes(activityTypes);
  config.validate();
  controller_->prepareTrace(config);
}

void ActivityProfilerProxy::startTrace() {
  controller_->startTrace();
}

std::unique_ptr<ActivityTraceInterface>
ActivityProfilerProxy::stopTrace() {
  return controller_->stopTrace();
}

bool ActivityProfilerProxy::isActive() {
  return controller_->isActive();
}

void ActivityProfilerProxy::pushCorrelationId(uint64_t id) {
  CuptiActivityInterface::pushCorrelationID(id,
    CuptiActivityInterface::CorrelationFlowType::Default);
}

void ActivityProfilerProxy::popCorrelationId() {
  CuptiActivityInterface::popCorrelationID(
    CuptiActivityInterface::CorrelationFlowType::Default);
}

void ActivityProfilerProxy::pushUserCorrelationId(uint64_t id) {
  CuptiActivityInterface::pushCorrelationID(id,
    CuptiActivityInterface::CorrelationFlowType::User);
}

void ActivityProfilerProxy::popUserCorrelationId() {
  CuptiActivityInterface::popCorrelationID(
    CuptiActivityInterface::CorrelationFlowType::User);
}

void ActivityProfilerProxy::transferCpuTrace(
   std::unique_ptr<CpuTraceBuffer> traceBuffer) {
  controller_->transferCpuTrace(std::move(traceBuffer));
}

bool ActivityProfilerProxy::enableForRegion(const std::string& match) {
  return controller_->traceInclusionFilter(match);
}

void ActivityProfilerProxy::addMetadata(
    const std::string& key, const std::string& value) {
  controller_->addMetadata(key, value);
}

void ActivityProfilerProxy::recordThreadInfo() {
  controller_->recordThreadInfo();
}

} // namespace libkineto
