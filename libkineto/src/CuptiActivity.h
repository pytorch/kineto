/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ITraceActivity.h"
#include "GenericTraceActivity.h"
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
  explicit CuptiActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  // see [Note: Temp Libkineto Nanosecond]
  int64_t timestamp() const override {
#ifdef TMP_LIBKINETO_NANOSECOND
    return activity_.start;
#else
    return nsToUs(activity_.start);
#endif
  }

  // see [Note: Temp Libkineto Nanosecond]
  int64_t duration() const override {
#ifdef TMP_LIBKINETO_NANOSECOND
    return activity_.end - activity_.start;
#else
    return nsToUs(activity_.end - activity_.start);
#endif
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override {return 0;}
  int32_t getThreadId() const override {return 0;}
  const ITraceActivity* linkedActivity() const override {return linked_;}
  int flowType() const override {return kLinkAsyncCpuGpu;}
  int flowId() const override {return correlationId();}
  const T& raw() const {return activity_;}
  const TraceSpan* traceSpan() const override {return nullptr;}

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// CUpti_ActivityAPI - CUDA runtime activities
struct RuntimeActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit RuntimeActivity(
      const CUpti_ActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {return activity_.correlationId;}
  int64_t deviceId() const override {return processId();}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::CUDA_RUNTIME;}
  bool flowStart() const override;
  const std::string name() const override {return runtimeCbidName(activity_.cbid);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA driver activities
struct DriverActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit DriverActivity(
      const CUpti_ActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {return activity_.correlationId;}
  int64_t deviceId() const override {return processId();}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::CUDA_DRIVER;}
  bool flowStart() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA runtime activities
struct OverheadActivity : public CuptiActivity<CUpti_ActivityOverhead> {
  explicit OverheadActivity(
      const CUpti_ActivityOverhead* activity,
      const ITraceActivity* linked,
      int32_t threadId=0)
      : CuptiActivity(activity, linked), threadId_(threadId) {}

  // see [Note: Temp Libkineto Nanosecond]
  int64_t timestamp() const override {
#ifdef TMP_LIBKINETO_NANOSECOND
    return activity_.start;
#else
    return nsToUs(activity_.start);
#endif
  }
  // see [Note: Temp Libkineto Nanosecond]
  int64_t duration() const override {
#ifdef TMP_LIBKINETO_NANOSECOND
    return activity_.end - activity_.start;
#else
    return nsToUs(activity_.end - activity_.start);
#endif
  }
  // TODO: Update this with PID ordering
  int64_t deviceId() const override {return -1;}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::OVERHEAD;}
  bool flowStart() const override;
  const std::string name() const override {return overheadKindString(activity_.overheadKind);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivitySynchronization - CUDA synchronization events
struct CudaSyncActivity : public CuptiActivity<CUpti_ActivitySynchronization> {
  explicit CudaSyncActivity(
      const CUpti_ActivitySynchronization* activity,
      const ITraceActivity* linked,
      int32_t srcStream,
      int32_t srcCorrId)
      : CuptiActivity(activity, linked),
        srcStream_(srcStream),
        srcCorrId_(srcCorrId) {}
  int64_t correlationId() const override {return raw().correlationId;}
  int64_t deviceId() const override;
  int64_t resourceId() const override;
  ActivityType type() const override {return ActivityType::CUDA_SYNC;}
  bool flowStart() const override {return false;}
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const CUpti_ActivitySynchronization& raw() const {return CuptiActivity<CUpti_ActivitySynchronization>::raw();}

 private:
  const int32_t srcStream_;
  const int32_t srcCorrId_;
};


// Base class for GPU activities.
// Can also be instantiated directly.
template<class T>
struct GpuActivity : public CuptiActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity* linked)
      : CuptiActivity<T>(activity, linked) {}
  int64_t correlationId() const override {return raw().correlationId;}
  int64_t deviceId() const override {return raw().deviceId;}
  int64_t resourceId() const override {return raw().streamId;}
  ActivityType type() const override;
  bool flowStart() const override {return false;}
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {return CuptiActivity<T>::raw();}
};

} // namespace KINETO_NAMESPACE
