/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivity.h"

#include <fmt/format.h>

#include "CudaDeviceProperties.h"
#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

// forward declaration
uint32_t contextIdtoDeviceId(uint32_t contextId);

template<>
inline const std::string GpuActivity<CUpti_ActivityKernel4>::name() const {
  return demangle(raw().name);
}

template<>
inline ActivityType GpuActivity<CUpti_ActivityKernel4>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

inline bool isWaitEventSync(CUpti_ActivitySynchronizationType type) {
  return (type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline bool isEventSync(CUpti_ActivitySynchronizationType type) {
  return (
    type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE ||
    type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline std::string eventSyncInfo(
    const CUpti_ActivitySynchronization& act,
    int32_t srcStream,
    int32_t srcCorrId
    ) {
  return fmt::format(R"JSON(
      "wait_on_stream": {},
      "wait_on_cuda_event_record_corr_id": {},
      "wait_on_cuda_event_id": {},)JSON",
      srcStream,
      srcCorrId,
      act.cudaEventId
  );
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

template<class T>
inline void GpuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

template<>
inline const std::string GpuActivity<CUpti_ActivityKernel4>::metadataJson() const {
  const CUpti_ActivityKernel4& kernel = raw();
  float blocksPerSmVal = blocksPerSm(kernel);
  float warpsPerSmVal = warpsPerSm(kernel);

  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  return fmt::format(R"JSON(
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "registers per thread": {},
      "shared memory": {},
      "blocks per SM": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "est. achieved occupancy %": {})JSON",
#ifdef TMP_LIBKINETO_NANOSECOND
      kernel.queued, kernel.deviceId, kernel.contextId,
#else
      us(kernel.queued), kernel.deviceId, kernel.contextId,
#endif
      kernel.streamId, kernel.correlationId,
      kernel.registersPerThread,
      kernel.staticSharedMemory + kernel.dynamicSharedMemory,
      std::isinf(blocksPerSmVal) ? "\"inf\"" : std::to_string(blocksPerSmVal),
      std::isinf(warpsPerSmVal) ? "\"inf\"" : std::to_string(warpsPerSmVal),
      kernel.gridX, kernel.gridY, kernel.gridZ,
      kernel.blockX, kernel.blockY, kernel.blockZ,
      (int) (0.5 + kernelOccupancy(kernel) * 100.0));
  // clang-format on
}


inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
  return fmt::format(
      "Memcpy {} ({} -> {})",
      memcpyKindString((CUpti_ActivityMemcpyKind)kind),
      memoryKindString((CUpti_ActivityMemoryKind)src),
      memoryKindString((CUpti_ActivityMemoryKind)dst));
}

template<>
inline ActivityType GpuActivity<CUpti_ActivityMemcpy>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemcpy>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemcpy>::metadataJson() const {
  const CUpti_ActivityMemcpy& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}


template<>
inline ActivityType GpuActivity<CUpti_ActivityMemcpy2>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemcpy2>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemcpy2>::metadataJson() const {
  const CUpti_ActivityMemcpy2& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "fromContext": {}, "inContext": {}, "toContext": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
      memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemset>::name() const {
  const char* memory_kind =
    memoryKindString((CUpti_ActivityMemoryKind)raw().memoryKind);
  return fmt::format("Memset ({})", memory_kind);
}

template<>
inline ActivityType GpuActivity<CUpti_ActivityMemset>::type() const {
  return ActivityType::GPU_MEMSET;
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemset>::metadataJson() const {
  const CUpti_ActivityMemset& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId,
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
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
  bool should_correlate =
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
      (activity_.cbid >= CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
       activity_.cbid <= CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020) ||
      activity_.cbid ==
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
      activity_.cbid ==
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000 ||
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020 ||
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 ||
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020;

#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
  should_correlate |=
      activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060;
#endif
  return should_correlate;
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(R"JSON(
      "cbid": {}, "correlation": {})JSON",
      activity_.cbid, activity_.correlationId);
}

inline bool DriverActivity::flowStart() const {
  return activity_.cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
}

inline const std::string DriverActivity::metadataJson() const {
  return fmt::format(R"JSON(
      "cbid": {}, "correlation": {})JSON",
      activity_.cbid, activity_.correlationId);
}

inline const std::string DriverActivity::name() const {
  // currently only cuLaunchKernel is expected
  assert(activity_.cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  return "cuLaunchKernel";
}

template<class T>
inline const std::string GpuActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
