/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

#include <fmt/format.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ApproximateClock.h"
#include "CuptiCbidRegistry.h"
#include "Demangle.h"
#include "DeviceProperties.h"
#include "GenericTraceActivity.h"
#include "ITraceActivity.h"
#include "Logger.h"
#include "ThreadUtil.h"
#include "cupti_strings.h"
#include "output_base.h"

namespace libkineto {
class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

// This function allows us to activate/deactivate TSC CUPTI callbacks
// via a killswitch
bool& use_cupti_tsc();

// These classes wrap the various CUPTI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Cupti activity type
template <class T>
struct CuptiActivity : public ITraceActivity {
  explicit CuptiActivity(const T* activity, const ITraceActivity* linked) : activity_(*activity), linked_(linked) {}
  // If we are running on Windows or are on a CUDA version < 11.6,
  // we use the default system clock so no conversion needed same for all
  // ifdefs below
  int64_t timestamp() const override {
#if defined(_WIN32) || CUDA_VERSION < 11060
    return activity_.start;
#else
    if (use_cupti_tsc()) {
      return get_time_converter()(activity_.start);
    } else {
      return activity_.start;
    }
#endif
  }

  int64_t duration() const override {
#if defined(_WIN32) || CUDA_VERSION < 11060
    return activity_.end - activity_.start;
#else
    if (use_cupti_tsc()) {
      return get_time_converter()(activity_.end) - get_time_converter()(activity_.start);
    } else {
      return activity_.end - activity_.start;
    }
#endif
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override {
    return 0;
  }
  int32_t getThreadId() const override {
    return 0;
  }
  const ITraceActivity* linkedActivity() const override {
    return linked_;
  }
  int flowType() const override {
    return kLinkAsyncCpuGpu;
  }
  int64_t flowId() const override {
    return correlationId();
  }
  const T& raw() const {
    return activity_;
  }
  const TraceSpan* traceSpan() const override {
    return nullptr;
  }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// CUpti_ActivityAPI - CUDA runtime activities
struct RuntimeActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit RuntimeActivity(const CUpti_ActivityAPI* activity, const ITraceActivity* linked, int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_.correlationId;
  }
  int64_t deviceId() const override {
    return processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::CUDA_RUNTIME;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return runtimeCbidName(activity_.cbid);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA driver activities
struct DriverActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit DriverActivity(const CUpti_ActivityAPI* activity, const ITraceActivity* linked, int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_.correlationId;
  }
  int64_t deviceId() const override {
    return processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::CUDA_DRIVER;
  }
  bool flowStart() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA runtime activities
struct OverheadActivity : public CuptiActivity<CUpti_ActivityOverhead> {
  explicit OverheadActivity(const CUpti_ActivityOverhead* activity, const ITraceActivity* linked, int32_t threadId = 0)
      : CuptiActivity(activity, linked), threadId_(threadId) {}

  int64_t timestamp() const override {
#if defined(_WIN32) || CUDA_VERSION < 11060
    return activity_.start;
#else
    if (use_cupti_tsc()) {
      return get_time_converter()(activity_.start);
    } else {
      return activity_.start;
    }
#endif
  }

  int64_t duration() const override {
#if defined(_WIN32) || CUDA_VERSION < 11060
    return activity_.end - activity_.start;
#else
    if (use_cupti_tsc()) {
      return get_time_converter()(activity_.end) - get_time_converter()(activity_.start);
    } else {
      return activity_.end - activity_.start;
    }
#endif
  }

  // TODO: Update this with PID ordering
  int64_t deviceId() const override {
    return -1;
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::OVERHEAD;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return overheadKindString(activity_.overheadKind);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivitySynchronization - CUDA synchronization events
struct CudaSyncActivity : public CuptiActivity<CUpti_ActivitySynchronization> {
  explicit CudaSyncActivity(const CUpti_ActivitySynchronization* activity,
                            const ITraceActivity* linked,
                            int32_t srcStream,
                            int32_t srcCorrId)
      : CuptiActivity(activity, linked), srcStream_(srcStream), srcCorrId_(srcCorrId) {}
  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override;
  int64_t resourceId() const override;
  ActivityType type() const override {
    return ActivityType::CUDA_SYNC;
  }
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const CUpti_ActivitySynchronization& raw() const {
    return CuptiActivity<CUpti_ActivitySynchronization>::raw();
  }

 private:
  const int32_t srcStream_;
  const int32_t srcCorrId_;
};

// Use CUpti_ActivityCudaEvent2 in CUDA 12.8+ for enhanced event tracking
#if CUDA_VERSION >= 12080
using CUpti_ActivityCudaEventType = CUpti_ActivityCudaEvent2;
#else
using CUpti_ActivityCudaEventType = CUpti_ActivityCudaEvent;
#endif

// Template specialization for timestamp() and duration() for CudaEvent types
template <>
inline int64_t CuptiActivity<CUpti_ActivityCudaEventType>::timestamp() const {
#if CUDA_VERSION >= 12080
#if defined(_WIN32)
  return activity_.deviceTimestamp;
#else
  if (use_cupti_tsc()) {
    return get_time_converter()(activity_.deviceTimestamp);
  } else {
    return activity_.deviceTimestamp;
  }
#endif
#else
  // For CUDA < 12.8, deviceTimestamp doesn't exist, set to 0
  return 0;
#endif
}

template <>
inline int64_t CuptiActivity<CUpti_ActivityCudaEventType>::duration() const {
  // Set duration to 1 to make events visible in trace
  return 1;
}

// CUpti_ActivityCudaEvent - CUDA event activities
struct CudaEventActivity : public CuptiActivity<CUpti_ActivityCudaEventType> {
  explicit CudaEventActivity(const CUpti_ActivityCudaEventType* activity, const ITraceActivity* linked)
      : CuptiActivity(activity, linked) {}

  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override;
  int64_t resourceId() const override;
  ActivityType type() const override {
    return ActivityType::CUDA_EVENT;
  }
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const CUpti_ActivityCudaEventType& raw() const {
    return CuptiActivity<CUpti_ActivityCudaEventType>::raw();
  }
};

// Base class for GPU activities.
// Can also be instantiated directly.
template <class T>
struct GpuActivity : public CuptiActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity* linked) : CuptiActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override {
    return raw().deviceId;
  }
  int64_t resourceId() const override {
    return raw().streamId;
  }
  ActivityType type() const override;
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {
    return CuptiActivity<T>::raw();
  }
};

// forward declaration
uint32_t contextIdtoDeviceId(uint32_t contextId);

template <>
inline const std::string GpuActivity<CUpti_ActivityKernelType>::name() const {
  return demangle(raw().name);
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityKernelType>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

inline bool isWaitEventSync(CUpti_ActivitySynchronizationType type) {
  return (type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline bool isEventSync(CUpti_ActivitySynchronizationType type) {
  return (type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE ||
          type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline std::string eventSyncInfo(const CUpti_ActivitySynchronization& act, int32_t srcStream, int32_t srcCorrId) {
  return fmt::format(
      R"JSON(
      "wait_on_stream": {},
      "wait_on_cuda_event_record_corr_id": {},
      "wait_on_cuda_event_id": {},)JSON",
      srcStream,
      srcCorrId,
      act.cudaEventId);
}

inline const std::string CudaSyncActivity::name() const {
  return syncTypeString(raw().type);
}

inline int64_t CudaSyncActivity::deviceId() const {
  return contextIdtoDeviceId(raw().contextId);
}

inline int64_t CudaSyncActivity::resourceId() const {
  // For Context and Device Sync events stream ID is invalid and
  // set to CUPTI_SYNCHRONIZATION_INVALID_VALUE (-1)
  // converting to an integer will automatically wrap the number to -1
  // in the trace.
  return static_cast<int32_t>(raw().streamId);
}

inline void CudaSyncActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline const std::string CudaSyncActivity::metadataJson() const {
  const CUpti_ActivitySynchronization& sync = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "cuda_sync_kind": "{}",{}
      "stream": {}, "correlation": {},
      "device": {}, "context": {})JSON",
      syncTypeString(sync.type),
      isEventSync(raw().type) ? eventSyncInfo(raw(), srcStream_, srcCorrId_) : "",
      static_cast<int32_t>(sync.streamId),
      sync.correlationId,
      deviceId(),
      sync.contextId);
  // clang-format on
  return "";
}

inline const std::string CudaEventActivity::name() const {
  return "cudaEvent";
}

inline int64_t CudaEventActivity::deviceId() const {
  return contextIdtoDeviceId(raw().contextId);
}

inline int64_t CudaEventActivity::resourceId() const {
  return static_cast<int32_t>(raw().streamId);
}

inline void CudaEventActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline const std::string CudaEventActivity::metadataJson() const {
  const CUpti_ActivityCudaEventType& event = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "event_id": {},
      "stream": {}, "correlation": {},
      "device": {}, "context": {})JSON",
      event.eventId,
      static_cast<int32_t>(event.streamId),
      event.correlationId,
      deviceId(),
      event.contextId);
  // clang-format on
}

template <class T>
inline void GpuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

template <class T>
inline std::string getGraphNodeMetadata(const T& activity) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  return fmt::format(
      R"JSON(,
      "graph id": {}, "graph node id": {})JSON",
      activity.graphId,
      activity.graphNodeId);
#else
  return "";
#endif
}

template <>
inline const std::string GpuActivity<CUpti_ActivityKernelType>::metadataJson() const {
  const CUpti_ActivityKernelType& kernel = raw();
  float blocksPerSmVal = blocksPerSm(kernel);
  float warpsPerSmVal = warpsPerSm(kernel);

  // clang-format off

  return fmt::format(R"JSON(
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "registers per thread": {},
      "shared memory": {},
      "blocks per SM": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "est. achieved occupancy %": {}{})JSON",
      kernel.queued, kernel.deviceId, kernel.contextId,
      kernel.streamId, kernel.correlationId,
      kernel.registersPerThread,
      kernel.staticSharedMemory + kernel.dynamicSharedMemory,
      std::isinf(blocksPerSmVal) ? "\"inf\"" : std::to_string(blocksPerSmVal),
      std::isinf(warpsPerSmVal) ? "\"inf\"" : std::to_string(warpsPerSmVal),
      kernel.gridX, kernel.gridY, kernel.gridZ,
      kernel.blockX, kernel.blockY, kernel.blockZ,
      (int) (0.5 + (kernelOccupancy(kernel) * 100.0)),
      getGraphNodeMetadata(kernel)
      );
  // clang-format on
}

inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
  return fmt::format("Memcpy {} ({} -> {})",
                     memcpyKindString((CUpti_ActivityMemcpyKind)kind),
                     memoryKindString((CUpti_ActivityMemoryKind)src),
                     memoryKindString((CUpti_ActivityMemoryKind)dst));
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemcpyType>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyType>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyType>::metadataJson() const {
  const CUpti_ActivityMemcpyType& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}{})JSON",
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, duration()),
      getGraphNodeMetadata(memcpy));
  // clang-format on
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemcpyPtoPType>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyPtoPType>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyPtoPType>::metadataJson() const {
  const CUpti_ActivityMemcpyPtoPType& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "fromContext": {}, "inContext": {}, "toContext": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}{})JSON",
      memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
      memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, duration()),
      getGraphNodeMetadata(memcpy));
  // clang-format on
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemsetType>::name() const {
  const char* memory_kind = memoryKindString((CUpti_ActivityMemoryKind)raw().memoryKind);
  return fmt::format("Memset ({})", memory_kind);
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemsetType>::type() const {
  return ActivityType::GPU_MEMSET;
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemsetType>::metadataJson() const {
  const CUpti_ActivityMemsetType& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {}{})JSON",
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId,
      memset.bytes, bandwidth(memset.bytes, duration()),
      getGraphNodeMetadata(memset));
  // clang-format on
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void DriverActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void OverheadActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline bool OverheadActivity::flowStart() const {
  return false;
}

inline const std::string OverheadActivity::metadataJson() const {
  return "";
}

inline bool RuntimeActivity::flowStart() const {
  return CuptiCbidRegistry::instance().requiresFlowCorrelation(CallbackDomain::RUNTIME, activity_.cbid);
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(
      R"JSON(
      "cbid": {}, "correlation": {})JSON",
      activity_.cbid,
      activity_.correlationId);
}

inline bool isTrackedDriverCbid(const CUpti_ActivityAPI& activity_) {
  return CuptiCbidRegistry::instance().isRegistered(CallbackDomain::DRIVER, activity_.cbid);
}

inline bool DriverActivity::flowStart() const {
  return CuptiCbidRegistry::instance().requiresFlowCorrelation(CallbackDomain::DRIVER, activity_.cbid);
}

inline const std::string DriverActivity::metadataJson() const {
  return fmt::format(
      R"JSON(
      "cbid": {}, "correlation": {})JSON",
      activity_.cbid,
      activity_.correlationId);
}

inline const std::string DriverActivity::name() const {
  return CuptiCbidRegistry::instance().getName(CallbackDomain::DRIVER, activity_.cbid);
}

template <class T>
inline const std::string GpuActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
