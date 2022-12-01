/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfiler.h"

#include "time_since_epoch.h"
#include "Config.h"
#include "ConfigLoader.h"
#include "ActivityProfilerThread.h"
#include "output_membuf.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

ActivityProfilerSession::ActivityProfilerSession(
    const Config& config, ActivityLoggerFactory& loggerFactory)
    : config_(config.clone()), loggerFactory_(loggerFactory) {
  // Add a LoggerObserverto collect all logs during the trace.
  logCollector_ = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(logCollector_.get());
}

ActivityProfilerSession::~ActivityProfilerSession() {
  Logger::removeLoggerObserver(logCollector_.get());
}

const std::vector<const ITraceActivity*>*
ActivityProfilerSession::activities() {
  if (!memLogger_) {
    auto mem_logger = std::make_unique<MemoryTraceLogger>(*config_);
    log(*mem_logger.get());
    memLogger_ = std::move(mem_logger);
  }
  return memLogger_->traceActivities();
}

void ActivityProfilerSession::save(const std::string& url) {
  std::string prefix;
  // if no protocol is specified, default to file
  if (url.find("://") == url.npos) {
    prefix = "file://";
  }
  log(*loggerFactory_.makeLogger(prefix + url));
}

void ActivityProfilerSession::log(ActivityLogger& logger) {
  if (status_ != TraceStatus::PROCESSING) {
    LOG(ERROR) << "log called with invalid TraceStatus: " << (int) status_;
    return;
  }
  if (memLogger_) {
    memLogger_->log(logger);
    return;
  }
  int32_t pid = processId();
  std::string process_name = processName(pid);
  if (process_name.empty()) {
    process_name = "Unknown";
  }
  recordDeviceInfo(pid, process_name, "CPU");

  // Log metadata
  logger.handleTraceStart(metadata_);

  // Activities
  for (const auto& kv : children_) {
    IActivityProfilerSession& child = *kv.second.get();
    if (child.status() == TraceStatus::PROCESSING) {
      child.log(logger);
    } else {
      LOG(WARNING) << "Not logging child session " << kv.first
                   << ", status = " << (int) child.status();
    }
  }

  // Device info
  for (const auto& kv : deviceInfo_) {
    logger.handleDeviceInfo(kv.second, startTime_);
  }

  // Resource Info
  for (const auto& kv : resourceInfo_) {
    logger.handleResourceInfo(kv.second, startTime_);
  }

  auto log_messages = logCollector_->extractCollectorMetadata();
  std::unordered_map<std::string, std::vector<std::string>> metadata;
  for (auto& md : log_messages) {
    metadata[toString(md.first)] = md.second;
  }
  logger.finalizeTrace(*config_, endTime_, metadata);
  LOG(INFO) << "Finished logging";
}

IActivityProfilerSession* ActivityProfilerSession::session(
    const std::string& name) const {
  LOG(INFO) << "session " << name;
  const auto& kv = children_.find(name);
  if (kv != children_.end()) {
    return kv->second.get();
  }
  LOG(INFO) << "Not found!";
  return nullptr;
}

ActivityProfiler::ActivityProfiler(
    const std::string name, IConfigLoader& configLoader)
    : name_(name), configLoader_(configLoader) {
  configLoader_.addHandler(ConfigLoader::ConfigKind::ActivityProfiler, this);
}

ActivityProfiler::~ActivityProfiler() {
  std::lock_guard<std::mutex> guard(mutex_);
  configLoader_.removeHandler(
      ConfigLoader::ConfigKind::ActivityProfiler, this);
}

void ActivityProfiler::registerProfiler(
    const std::string& name,
    const ActivityProfilerFactory::Creator creator) {
  if (delegates_.find(name) == delegates_.end()) {
    profilerFactory_.addCreator(name, creator);
    delegates_[name] = nullptr;
  } else {
    throw std::invalid_argument("Profiler already registered");
  }
}

void ActivityProfiler::init(ICompositeProfiler* /* unused */) {
  // Create and initialize profilers
  supportedActivityTypes_.clear();
  for (auto& kv : delegates_) {
    const std::string& name = kv.first;
    auto& profiler = kv.second;
    if (!profiler) {
      profiler = profilerFactory_.create(name);
    }
    if (!profiler->isInitialized()) {
      profiler->init(this);
    }
    const auto& activities = profiler->supportedActivityTypes();
    supportedActivityTypes_.insert(activities.begin(), activities.end());
  }
}

std::future<std::shared_ptr<IProfilerSession>>
ActivityProfiler::acceptConfig(const Config& config) {
  LOG(INFO) << "acceptConfig";
  std::promise<std::shared_ptr<IProfilerSession>> promise;
  auto res = promise.get_future();
  if (config.activityProfilerEnabled()) {
    LOG(INFO) << "scheduleTrace";
    VLOG(1) << "scheduleTrace";
    // This handler should return quickly -
    // start a profilerLoop() thread to handle request
    std::lock_guard<std::mutex> guard(mutex_);
    if (!profilerThread_ || !profilerThread_->active()) {
      profilerThread_.reset(new ActivityProfilerThread(*this, config, std::move(promise)));
    } else {
      LOG(ERROR) << "Prior trace request in progress";
      promise.set_value(nullptr);
    }
  } else {
    promise.set_value(nullptr);
  }
  return res;
}

std::shared_ptr<IActivityProfilerSession> ActivityProfiler::configure(
    const Config& options,
    ICompositeProfilerSession* /*unused*/) {
  // Current async request will be cancelled
  if (profilerThread_ && !profilerThread_->isCurrentThread()) {
    profilerThread_ = nullptr;
    stop(*session_);
  }
  auto session = std::make_shared<ActivityProfilerSession>(
      options, loggerFactory_);
  for (const auto& kv : delegates_) {
    // More than one profiler may generate a given activity type
    // in which case we invoke all
    const auto& profiler = kv.second;
    if (!profiler) {
      LOG(ERROR) << "Profiler not initialized: " << kv.first;
    }
    const auto& supported = profiler->supportedActivityTypes();
    for (const ActivityType type : options.selectedActivityTypes()) {
      if (supported.find(type) != supported.end()) {
        LOG(INFO) << "add child " << profiler->name();
        session->addChild(
            profiler->name(),
            profiler->configure(options, session.get()));
        break;
      }
    }
  }

  session_ = session;
  session->status(TraceStatus::WARMUP);
  return session;
}

void ActivityProfiler::start(IActivityProfilerSession& session) {
  if (session.status() != TraceStatus::WARMUP) {
    LOG(ERROR) << "start called with invalid status: " << (int) session.status();
    session.status(TraceStatus::ERROR);
    return;
  }
  ActivityProfilerSession& composite_session =
      dynamic_cast<ActivityProfilerSession&>(session);
  composite_session.startTime(timeSinceEpoch());
  for (auto& kv : delegates_) {
    if (auto* child_session = composite_session.session(kv.first)) {
      kv.second->start(*child_session);
    }
  }
  session.status(TraceStatus::RECORDING);
}

void ActivityProfiler::stop(IActivityProfilerSession& session) {
  if (session.status() != TraceStatus::WARMUP &&
      session.status() != TraceStatus::RECORDING) {
    LOG(ERROR) << "stop called with invalid status: " << (int) session.status();
    session.status(TraceStatus::ERROR);
  }
  ActivityProfilerSession& composite_session =
      dynamic_cast<ActivityProfilerSession&>(session);
  composite_session.endTime(timeSinceEpoch());
  for (auto& kv : delegates_) {
    if (auto* child_session = composite_session.session(kv.first)) {
      kv.second->stop(*child_session);
    }
  }
  session.status(TraceStatus::PROCESSING);
}

} // namespace libkineto

