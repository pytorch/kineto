/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <roctracer.h>
#include <roctracer_hip.h>
#include <roctracer_ext.h>
#include <roctracer_roctx.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ITraceActivity.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "RoctracerLogger.h"

namespace libkineto {
  class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

// These classes wrap the various Roctracer activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Roctracer activity type
template<class T>
struct RoctracerActivity : public ITraceActivity {
  explicit RoctracerActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  // Our stored timestamps (from roctracer and generated) are in CLOCK_MONOTONIC domain (in ns).
  // Convert the timestamps.
  int64_t timestamp() const override {
    return activity_.begin;
  }
  int64_t duration() const override {
    return activity_.end - activity_.begin;
  }
  int64_t correlationId() const override {return 0;}
  int32_t getThreadId() const override {return 0;}
  const ITraceActivity* linkedActivity() const override {return linked_;}
  int flowType() const override {return kLinkAsyncCpuGpu;}
  int flowId() const override {return correlationId();}
  const T& raw() const {return activity_;}
  const TraceSpan* traceSpan() const override {return nullptr;}
  const std::string getMetadataValue(const std::string& key) const override {
    auto it = metadata_.find(key);
    if (it != metadata_.end()) {
      return it->second;
    }
    return "";
  }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
  std::unordered_map<std::string, std::string> metadata_;
};

// roctracerAsyncRow - Roctracer GPU activities
struct GpuActivity : public RoctracerActivity<roctracerAsyncRow> {
  explicit GpuActivity(
      const roctracerAsyncRow* activity,
      const ITraceActivity* linked)
      : RoctracerActivity(activity, linked) {
    switch (activity_.kind) {
      case HIP_OP_COPY_KIND_DEVICE_TO_HOST_:
      case HIP_OP_COPY_KIND_HOST_TO_DEVICE_:
      case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_:
      case HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_:
      case HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_:
      case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_:
        type_ = ActivityType::GPU_MEMCPY;
        break;
      case HIP_OP_COPY_KIND_FILL_BUFFER_:
        type_ = ActivityType::GPU_MEMSET;
        break;
      case HIP_OP_DISPATCH_KIND_KERNEL_:
      case HIP_OP_DISPATCH_KIND_TASK_:
      default:
        type_ = ActivityType::CONCURRENT_KERNEL;
        break;
    }
  }
  int64_t correlationId() const override {return activity_.id;}
  int64_t deviceId() const override {return activity_.device;}
  int64_t resourceId() const override {return  activity_.queue;}
  ActivityType type() const override {return type_;};
  bool flowStart() const override {return false;}
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
   ActivityType type_;
};

// roctracerRow, roctracerKernelRow, roctracerCopyRow, roctracerMallocRow - Roctracer runtime activities
template <class T>
struct RuntimeActivity : public RoctracerActivity<T> {
  explicit RuntimeActivity(
      const T* activity,
      const ITraceActivity* linked)
      : RoctracerActivity<T>(activity, linked) {}
  int64_t correlationId() const override {return raw().id;}
  int64_t deviceId() const override {return raw().pid;}
  int64_t resourceId() const override {return raw().tid;}
  ActivityType type() const override {return ActivityType::CUDA_RUNTIME;}
  bool flowStart() const override;
  const std::string name() const override {return std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, raw().cid, 0));}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {return RoctracerActivity<T>::raw();}
};

} // namespace KINETO_NAMESPACE

// Include the implementation detail of this header file.
// The *_inl.h helps separate header interface from implementation details.
#include "RoctracerActivity_inl.h"
