/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "ActivityLoggerFactory.h"
#include "ActivityProfilerFactory.h"
#include "ActivityType.h"
#include "IActivityProfiler.h"
#include "ICorrelationObserver.h"
#include "ResourceInfo.h"

namespace libkineto {

class Config;

class ICompositeProfilerSession
    : public IActivityProfilerSession, public ICorrelationObserver {
 public:
  virtual void registerCorrelationObserver(ICorrelationObserver* observer) = 0;

  virtual void recordDeviceInfo(
      int64_t device, std::string name, std::string label) = 0;

  virtual DeviceInfo deviceInfo(int64_t id) const = 0;

  // Saves information for the current thread to be used in profiler output
  // Client must record any new kernel thread where the activity has occured.
  virtual void recordThreadInfo() = 0;
  virtual void recordResourceInfo(
      int64_t device, int64_t id, int sort_index, std::string name) = 0;
  virtual void recordResourceInfo(
      int64_t device,
      int64_t id,
      int sort_index,
      std::function<std::string()> name) = 0;
  virtual ResourceInfo resourceInfo(int64_t device, int64_t resource) const = 0;

  virtual void addMetadata(const std::string& key, const std::string& value) = 0;

  virtual void addChild(
      const std::string& name,
      std::shared_ptr<IActivityProfilerSession> session) = 0;
  virtual IActivityProfilerSession* session(const std::string& name) const = 0;

  virtual int64_t startTime() = 0;
  virtual int64_t endTime() = 0;

  virtual const std::vector<const ITraceActivity*>* activities() = 0;
  virtual void save(const std::string& path) = 0;
};

class ICompositeProfiler : public IActivityProfiler {
 public:
  virtual void registerProfiler(
      const std::string& name,
      const ActivityProfilerFactory::Creator creator) = 0;

  virtual void registerLogger(
      const std::string& protocol,
      ActivityLoggerFactory::Creator creator) = 0;

  virtual ActivityLoggerFactory& loggerFactory() = 0;

  virtual IActivityProfiler* profiler(const std::string& name) = 0;

  virtual bool isProfilerRegistered(const std::string& name) = 0;
 };

} // namespace libkineto
