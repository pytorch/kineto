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
#include <unordered_map>
#include <vector>

#include "Config.h"
#include "GenericTraceActivity.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

class Config;

class MemoryTraceLogger : public ActivityLogger {
 public:
  MemoryTraceLogger(const Config& config) : config_(config.clone()) {
    activities_.reserve(100000);
  }

  // Note: the caller of these functions should handle concurrency
  // i.e., these functions are not thread-safe
  void handleDeviceInfo(
      const DeviceInfo& info,
      uint64_t time) override {
    deviceInfoList_.emplace_back(info, time);
  }

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override {
    resourceInfoList_.emplace_back(info, time);
  }

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override {}

  void handleTraceSpan(const TraceSpan& span) override {
    // Handled separately
  }

  template<class T>
  void addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    activities_.push_back(wrappers_.back().get());
  }

  // Just add the pointer to the list - ownership of the underlying
  // objects must be transferred in ActivityBuffers via finalizeTrace
  void handleActivity(const ITraceActivity& activity) override {
    activities_.push_back(&activity);
  }
  void handleGenericActivity(const GenericTraceActivity& activity) override {
    addActivityWrapper(activity);
  }

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override {
    metadata_ = metadata;
  }

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) override {
    buffers_ = std::move(buffers);
    endTime_ = endTime;
  }

  const std::vector<const ITraceActivity*>* traceActivities() {
    return &activities_;
  }

  void log(ActivityLogger& logger) {
    logger.handleTraceStart(metadata_);
    for (auto& activity : activities_) {
      activity->log(logger);
    }
    for (auto& p : deviceInfoList_) {
      logger.handleDeviceInfo(p.first, p.second);
    }
    for (auto& p : resourceInfoList_) {
      logger.handleResourceInfo(p.first, p.second);
    }
    for (auto& cpu_trace_buffer : buffers_->cpu) {
      logger.handleTraceSpan(cpu_trace_buffer->span);
    }
    // Hold on to the buffers
    logger.finalizeTrace(*config_, nullptr, endTime_, loggerMetadata_);
  }

  void setLoggerMetadata(
      std::unordered_map<std::string, std::vector<std::string>>&& lmd) {
    loggerMetadata_ = std::move(lmd);
  }

 private:

  std::unique_ptr<Config> config_;
  // Optimization: Remove unique_ptr by keeping separate vector per type
  std::vector<const ITraceActivity*> activities_;
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
  std::vector<std::pair<DeviceInfo, int64_t>> deviceInfoList_;
  std::vector<std::pair<ResourceInfo, int64_t>> resourceInfoList_;
  std::unique_ptr<ActivityBuffers> buffers_;
  std::unordered_map<std::string, std::string> metadata_;
  std::unordered_map<std::string, std::vector<std::string>> loggerMetadata_;
  int64_t endTime_{0};
};

} // namespace KINETO_NAMESPACE
