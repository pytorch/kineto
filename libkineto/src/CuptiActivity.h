 /*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

#include "ITraceActivity.h"
#include "CuptiActivityPlatform.h"
#include "ThreadUtil.h"
#include "cupti_strings.h"

namespace libkineto {
  class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

// These classes wrap the various CUPTI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Cupti activity type
template<class T>
struct CuptiActivity : public ITraceActivity {
  explicit CuptiActivity(const T* activity, const ITraceActivity& linked)
      : activity_(*activity), linked_(linked) {}
  int64_t timestamp() const override {
    return nsToUs(unixEpochTimestamp(activity_.start));
  }
  int64_t duration() const override {
    return nsToUs(activity_.end - activity_.start);
  }
  int64_t correlationId() const override {return activity_.correlationId;}
  int flowType() const override {return 0;}
  int flowId() const override {return 0;}
  const T& raw() const {return activity_;}
  const ITraceActivity* linkedActivity() const override {return &linked_;}
  const TraceSpan* traceSpan() const override {return nullptr;}

 protected:
  const T& activity_;
  const ITraceActivity& linked_;
};

// CUpti_ActivityAPI - CUDA runtime activities
struct RuntimeActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit RuntimeActivity(
      const CUpti_ActivityAPI* activity,
      const ITraceActivity& linked,
      int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t deviceId() const override {return processId();}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::CUDA_RUNTIME;}
  const std::string name() const override {return runtimeCbidName(activity_.cbid);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Base class for GPU activities.
// Can also be instantiated directly.
template<class T>
struct GpuActivity : public CuptiActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity& linked)
      : CuptiActivity<T>(activity, linked) {}
  int64_t deviceId() const override {return raw().deviceId;}
  int64_t resourceId() const override {return raw().streamId;}
  ActivityType type() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {return CuptiActivity<T>::raw();}
};

} // namespace KINETO_NAMESPACE
