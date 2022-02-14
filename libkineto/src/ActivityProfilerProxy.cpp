// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ActivityProfilerProxy.h"

#include "ActivityProfilerController.h"
#include "Config.h"
#include "CuptiActivityApi.h"
#include "Logger.h"
#include <chrono>

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

  // allow user provided config to override default options
  if (!configStr.empty()) {
    if (!config.parse(configStr)) {
      LOG(WARNING) << "Failed to parse config : " << configStr;
    }
    // parse also runs validate
    validate_required = false;
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
}

void ActivityProfilerProxy::popCorrelationId() {
  CuptiActivityApi::popCorrelationID(
    CuptiActivityApi::CorrelationFlowType::Default);
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

} // namespace libkineto
