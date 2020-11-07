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

#include <cupti.h>

#include "Config.h"
#include "ClientTraceActivity.h"
#include "CuptiActivity.h"
#include "CuptiActivity.tpp"
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
  void handleProcessInfo(
      const ProcessInfo& processInfo,
      uint64_t time) override {
    processInfoList_.emplace_back(processInfo, time);
  }

  void handleThreadInfo(const ThreadInfo& threadInfo, int64_t time) override {
    threadInfoList_.emplace_back(threadInfo, time);
  }

  void handleTraceSpan(const TraceSpan& span) override {
    traceSpanList_.push_back(span);
  }

  void handleIterationStart(const TraceSpan& span) override {
    iterationList_.push_back(span);
  }

  void handleCpuActivity(
      const libkineto::ClientTraceActivity& activity,
      const TraceSpan& span) override {
    activities_.push_back(
        std::make_unique<CpuActivityDecorator>(activity, span));
  }

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

  void finalizeTrace(const Config& config, std::unique_ptr<ActivityBuffers> buffers) override {
    buffers_ = std::move(buffers);
  }

  const std::vector<std::unique_ptr<TraceActivity>>* traceActivities() {
    return &activities_;
  }

  void log(ActivityLogger& logger) {
    for (auto& activity : activities_) {
      activity->log(logger);
    }
    for (auto& p : processInfoList_) {
      logger.handleProcessInfo(p.first, p.second);
    }
    for (auto& p : threadInfoList_) {
      logger.handleThreadInfo(p.first, p.second);
    }
    for (auto& span : traceSpanList_) {
      logger.handleTraceSpan(span);
    }
    for (auto& it : iterationList_) {
      logger.handleIterationStart(it);
    }
    finalizeTrace(*config_, std::move(buffers_));
  }

 private:

  struct CpuActivityDecorator : public libkineto::TraceActivity {
    CpuActivityDecorator(
        const libkineto::ClientTraceActivity& activity,
        const TraceSpan& span)
        : wrappee_(activity), span_(span) {}
    int64_t deviceId() const override {return wrappee_.deviceId();}
    int64_t resourceId() const override {return wrappee_.resourceId();}
    int64_t timestamp() const override {return wrappee_.timestamp();}
    int64_t duration() const override {return wrappee_.duration();}
    int64_t correlationId() const override {return wrappee_.correlationId();}
    ActivityType type() const override {return wrappee_.type();}
    const std::string name() const override {return wrappee_.name();}
    const TraceActivity* linkedActivity() const override {
      return wrappee_.linkedActivity();
    }
    void log(ActivityLogger& logger) const override {
      logger.handleCpuActivity(wrappee_, span_);
    }
    const libkineto::ClientTraceActivity& wrappee_;
    const TraceSpan span_;
  };

  std::unique_ptr<Config> config_;
  // Optimization: Remove unique_ptr by keeping separate vector per type
  std::vector<std::unique_ptr<TraceActivity>> activities_;
  std::vector<std::pair<ProcessInfo, int64_t>> processInfoList_;
  std::vector<std::pair<ThreadInfo, int64_t>> threadInfoList_;
  std::vector<TraceSpan> traceSpanList_;
  std::vector<TraceSpan> iterationList_;
  std::unique_ptr<ActivityBuffers> buffers_;
};

} // namespace KINETO_NAMESPACE
