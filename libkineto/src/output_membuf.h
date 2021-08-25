/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef HAS_CUPTI
#include <cupti.h>
#endif

#include "Config.h"
#include "GenericTraceActivity.h"
#ifdef HAS_CUPTI
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
#endif // HAS_CUPTI
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

  void handleTraceSpan(const TraceSpan& span) override {
    // Handled separately
  }

  void handleGenericActivity(const GenericTraceActivity& activity) override {
    activities_.push_back(
        std::make_unique<GenericTraceActivity>(activity));
  }

#ifdef HAS_CUPTI
  void handleRuntimeActivity(
      const RuntimeActivity& activity) override {
    activities_.push_back(std::make_unique<RuntimeActivity>(activity));
  }

  void handleGpuActivity(const GpuActivity<CUpti_ActivityKernel4>& activity) override {
    activities_.push_back(std::make_unique<GpuActivity<CUpti_ActivityKernel4>>(activity));
  }
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy>& activity) override {
    activities_.push_back(std::make_unique<GpuActivity<CUpti_ActivityMemcpy>>(activity));
  }
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemcpy2>& activity) override {
    activities_.push_back(std::make_unique<GpuActivity<CUpti_ActivityMemcpy2>>(activity));
  }
  void handleGpuActivity(const GpuActivity<CUpti_ActivityMemset>& activity) override {
    activities_.push_back(std::make_unique<GpuActivity<CUpti_ActivityMemset>>(activity));
  }
#endif // HAS_CUPTI

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override {
    metadata_ = metadata;
  }

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override {
    buffers_ = std::move(buffers);
    endTime_ = endTime;
  }

  const std::vector<std::unique_ptr<TraceActivity>>* traceActivities() {
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
    logger.finalizeTrace(*config_, nullptr, endTime_);
  }

 private:

  std::unique_ptr<Config> config_;
  // Optimization: Remove unique_ptr by keeping separate vector per type
  std::vector<std::unique_ptr<TraceActivity>> activities_;
  std::vector<std::pair<DeviceInfo, int64_t>> deviceInfoList_;
  std::vector<std::pair<ResourceInfo, int64_t>> resourceInfoList_;
  std::unique_ptr<ActivityBuffers> buffers_;
  std::unordered_map<std::string, std::string> metadata_;
  int64_t endTime_{0};
};

} // namespace KINETO_NAMESPACE
