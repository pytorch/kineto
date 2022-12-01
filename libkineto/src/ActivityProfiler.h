/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "ActivityType.h"
#include "ActivityProfilerFactory.h"
#include "IActivityProfiler.h"
#include "IActivityProfilerThread.h"
#include "ICompositeProfiler.h"
#include "IConfigHandler.h"
#include "ICorrelationObserver.h"
#include "ActivityLoggerFactory.h"
#include "LoggerCollector.h"
#include "ResourceInfo.h"

namespace libkineto {

class ActivityProfilerController;
class Config;
class IConfigLoader;
class MemoryTraceLogger;

class ActivityProfilerSession : public ICompositeProfilerSession {
 public:

  explicit ActivityProfilerSession(
      const Config& config, ActivityLoggerFactory& loggerFactory);
  virtual ~ActivityProfilerSession();

  void registerCorrelationObserver(ICorrelationObserver* observer) override {
    correlationObservers_.push_back(observer);
  }

  std::mutex& mutex() override {
    return mutex_;
  }

  TraceStatus status() override {
    return status_;
  }

  void status(TraceStatus status) override {
    status_ = status;
  }

  // returns list of Trace Activities
  const std::vector<const ITraceActivity*>* activities() override;

  // returns errors with this trace
  std::vector<std::string> errors() override {
    // FIXME
    return {};
  }

  void save(const std::string& url) override;

  // processes trace activities using logger
  void log(ActivityLogger& logger) override;

  void pushCorrelationId(ActivityType kind, uint64_t id) override {
    for (auto* observer : correlationObservers_) {
      observer->pushCorrelationId(kind, id);
    }
  }

  void popCorrelationId(ActivityType kind) override {
    for (auto* observer : correlationObservers_) {
      observer->popCorrelationId(kind);
    }
  }

  void recordDeviceInfo(
      int64_t device, std::string name, std::string label) override {
    deviceInfo_.emplace(device, DeviceInfo(device, name, label));
  }

  DeviceInfo deviceInfo(int64_t id) const override {
    const auto& it = deviceInfo_.find(id);
    return it == deviceInfo_.end()
        ? DeviceInfo(-1, "invalid", "") : it->second;
  }

  // Create resource names for streams
  void recordResourceInfo(
      int64_t device, int64_t id, int sort_index, std::string name) override {
    resourceInfo_.emplace(
        std::make_pair(device, id), ResourceInfo(device, id, sort_index, name));
  }

  void recordResourceInfo(
      int64_t device,
      int64_t id,
      int sort_index,
      std::function<std::string()> name) override {
    if (resourceInfo_.find({device, id}) == resourceInfo_.end()) {
      recordResourceInfo(device, id, sort_index, name());
    }
  }

  void recordThreadInfo() override {
    // Note we're using the lower 32 bits of the (opaque) pthread id
    // as key, because that's what CUPTI records.
    int32_t tid = threadId();
    recordResourceInfo(processId(), tid, tid, []() {
        return fmt::format(
            "thread {} ({})", systemThreadId(), getThreadName());
    });
  }

  ResourceInfo resourceInfo(int64_t device, int64_t resource) const override {
    const auto& it = resourceInfo_.find({device, resource});
    return it == resourceInfo_.end()
        ? ResourceInfo(-1, -1, -1, "invalid") : it->second;
  }

  void addMetadata(const std::string& key, const std::string& value) override {
    metadata_[key] = value;
  }

  void addChild(
      const std::string& name,
      std::shared_ptr<IActivityProfilerSession> session) override {
    children_[name] = std::move(session);
  }

  IActivityProfilerSession* session(const std::string& name) const override;

  int64_t startTime() override {
    return startTime_;
  }

  void startTime(int64_t t) {
    startTime_ = t;
  }

  int64_t endTime() override {
    return endTime_;
  }

  void endTime(int64_t t) {
    endTime_ = t;
  }

  // FIXME: Temporary hack until profilers have been properly separated
  void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) override {
    session("CuptiProfiler")->transferCpuTrace(std::move(cpuTrace));
  }

 private:
  std::mutex mutex_;
  TraceStatus status_{TraceStatus::READY};
  std::unique_ptr<Config> config_;
  std::map<std::string, std::shared_ptr<IActivityProfilerSession>> children_;
  std::vector<ICorrelationObserver*> correlationObservers_;

  std::map<int64_t, DeviceInfo> deviceInfo_;
  std::map<std::pair<int64_t, int64_t>, ResourceInfo> resourceInfo_;

  // Trace metadata
  std::unordered_map<std::string, std::string> metadata_;
  std::unique_ptr<LoggerCollector> logCollector_;

  std::unique_ptr<MemoryTraceLogger> memLogger_;
  ActivityLoggerFactory& loggerFactory_;

  int64_t startTime_{0};
  int64_t endTime_{0};
};

class ActivityProfiler
    : public ICompositeProfiler, IConfigHandler  {

 public:
  ActivityProfiler(
      const std::string name, IConfigLoader& cfgLoader);
  ~ActivityProfiler() override;

  const std::string name() const override {
    return name_;
  }

  void registerProfiler(
      const std::string& name,
      const ActivityProfilerFactory::Creator creator) override;

  void registerLogger(
      const std::string& protocol,
      ActivityLoggerFactory::Creator creator) override {
    loggerFactory_.addProtocol(protocol, creator);
  }

  ActivityLoggerFactory& loggerFactory() override {
    return loggerFactory_;
  }

  // IConfigHandler
  bool canAcceptConfig() override {
    return !isActive();
  }

  std::future<std::shared_ptr<IProfilerSession>>
  acceptConfig(const Config& config) override;

  void init(ICompositeProfiler* /* unused */) override;

  bool isInitialized() const override {
    // Return true if at least one profiler is initialized
    for (const auto& kv : delegates_) {
      const auto& delegate = kv.second;
      if (delegate && delegate->isInitialized()) {
        return true;
      }
    }
    return false;
  }

  const std::set<ActivityType>& supportedActivityTypes() const override {
    return supportedActivityTypes_;
  }

  bool isActive() const override {
    // Return true if at least one profiler is active
    for (const auto& kv : delegates_) {
      const auto& delegate = kv.second;
      if (delegate && delegate->isActive()) {
        return true;
      }
    }
    return false;
  }

  std::shared_ptr<IActivityProfilerSession> configure(
      const Config& options,
      ICompositeProfilerSession* parentSession) override;

  void start(IActivityProfilerSession& session) override;
  void stop(IActivityProfilerSession& session) override;

  IActivityProfiler* profiler(const std::string& name) override {
    const auto& kv = delegates_.find(name);
    if (kv != delegates_.end()) {
      return kv->second.get();
    }
    return nullptr;
  }

  bool isProfilerRegistered(const std::string& name) override {
    return delegates_.find(name) != delegates_.end();
  }

 private:
  std::string name_;
  IConfigLoader& configLoader_;
  ActivityProfilerFactory profilerFactory_;
  ActivityLoggerFactory loggerFactory_;
  std::map<std::string, std::unique_ptr<IActivityProfiler>> delegates_;
  std::set<ActivityType> supportedActivityTypes_;
  // Trace metadata
  std::unordered_map<std::string, std::string> metadata_;
  std::mutex mutex_;
  std::unique_ptr<IActivityProfilerThread> profilerThread_;
  std::shared_ptr<ActivityProfilerSession> session_;
  std::atomic_int nextCorrelationKind_{0};
};

} // namespace libkineto
