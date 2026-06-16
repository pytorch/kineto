/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "RocprofActivity.h"

#include <fmt/format.h>
#include <stddef.h>
#include <cstdint>
#include <string>
#include <vector>

#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

namespace {
thread_local std::unordered_map<int, std::vector<int64_t>> correlationToGrid;
thread_local std::unordered_map<int, std::vector<int64_t>> correlationToBlock;
thread_local std::unordered_map<int, size_t> correlationToSize;

inline std::vector<int64_t> rocprofKernelGrid(const rocprofKernelRow& activity) {
  return {
      static_cast<int64_t>(activity.gridX), static_cast<int64_t>(activity.gridY), static_cast<int64_t>(activity.gridZ)};
}

inline std::vector<int64_t> rocprofKernelBlock(const rocprofKernelRow& activity) {
  return {static_cast<int64_t>(activity.workgroupX),
          static_cast<int64_t>(activity.workgroupY),
          static_cast<int64_t>(activity.workgroupZ)};
}

inline std::string vectorToJsonArray(const std::vector<int64_t>& values) {
  return fmt::format("[{}, {}, {}]", values[0], values[1], values[2]);
}
} // namespace

inline const char* getGpuActivityKindString(uint32_t domain, uint32_t op) {
  if (domain == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
    return "Dispatch Kernel";
  else if (domain == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
    switch (op) {
      case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:
        return "HtoH";
      case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:
        return "HtoD";
      case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
        return "DtoH";
      case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
        return "DtoD";
    }
  }
  return "<unknown>";
}

inline void getMemcpySrcDstString(uint32_t kind, std::string& src, std::string& dst) {
  switch (kind) {
    case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:
      src = "Host";
      dst = "Host";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
      src = "Device";
      dst = "Host";
      break;
    case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:
      src = "Host";
      dst = "Device";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
      src = "Device";
      dst = "Device";
      break;
    default:
      src = "?";
      dst = "?";
      break;
  }
}

// GPU Activities

inline const std::string GpuActivity::name() const {
  if (type_ == ActivityType::CONCURRENT_KERNEL) {
    auto op = raw().op;
    auto domain = raw().domain;
    std::string opString = RocprofLogger::opString(static_cast<rocprofiler_buffer_tracing_kind_t>(domain), op);
    const char* name = opString.c_str();
    return demangle(raw().kernelName.length() > 0 ? raw().kernelName : std::string(name));
  } else if (type_ == ActivityType::GPU_MEMSET) {
    return fmt::format("Memset ({})", getGpuActivityKindString(raw().domain, raw().op));
  } else if (type_ == ActivityType::GPU_MEMCPY) {
    std::string src = "";
    std::string dst = "";
    getMemcpySrcDstString(raw().op, src, dst);
    return fmt::format("Memcpy {} ({} -> {})", getGpuActivityKindString(raw().domain, raw().op), src, dst);
  } else {
    return "";
  }
  return "";
}

inline void GpuActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

static inline std::string bandwidth(size_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

static inline void addBandwidthTypedMetadata(TypedMetadata& metadata, size_t bytes, uint64_t duration) {
  if (duration == 0) {
    return;
  }
  metadata.set(RocmMetadataFields::kMemoryBandwidthGbps, static_cast<double>(bytes) / static_cast<double>(duration));
}

inline const std::string GpuActivity::metadataJson() const {
  const auto& gpuActivity = raw();
  // clang-format off

  // if memcpy or memset, add size
  auto sizeIt = correlationToSize.find(gpuActivity.id);
  if (sizeIt != correlationToSize.end()) {
    size_t size = sizeIt->second;
    std::string bandwidth_gib = (bandwidth(size, gpuActivity.end - gpuActivity.begin));
    return fmt::format(R"JSON(
      "device": {}, "stream": {}, "hsa_queue": {},
      "correlation": {}, "kind": "{}",
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      gpuActivity.device, resourceId(), gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op),
      size, bandwidth_gib);
  }

  auto gridIt = correlationToGrid.find(gpuActivity.id);
  auto blockIt = correlationToBlock.find(gpuActivity.id);

  // if compute kernel, add grid and block
  if (gridIt != correlationToGrid.end() && blockIt != correlationToBlock.end()) {
    return fmt::format(R"JSON(
      "device": {}, "stream": {}, "hsa_queue": {},
      "correlation": {}, "kind": "{}",
      "grid": {}, "block": {})JSON",
      gpuActivity.device, resourceId(), gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op),
      vectorToJsonArray(gridIt->second), vectorToJsonArray(blockIt->second));
  } else {
    return fmt::format(R"JSON(
      "device": {}, "stream": {}, "hsa_queue": {},
      "correlation": {}, "kind": "{}")JSON",
      gpuActivity.device, resourceId(), gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op));
  }
  // clang-format on
}

inline TypedMetadata GpuActivity::typedMetadata() const {
  const auto& gpuActivity = raw();
  TypedMetadata metadata;
  metadata.set(RocmMetadataFields::kDevice, static_cast<int64_t>(gpuActivity.device));
  metadata.set(RocmMetadataFields::kStream, resourceId());
  metadata.set(RocmMetadataFields::kHsaQueue, static_cast<int64_t>(gpuActivity.queue));
  metadata.set(RocmMetadataFields::kCorrelation, static_cast<int64_t>(gpuActivity.id));
  metadata.set(RocmMetadataFields::kKind, std::string{getGpuActivityKindString(gpuActivity.domain, gpuActivity.op)});

  // if memcpy or memset, add size
  auto sizeIt = correlationToSize.find(gpuActivity.id);
  if (sizeIt != correlationToSize.end()) {
    metadata.set(RocmMetadataFields::kBytes, static_cast<int64_t>(sizeIt->second));
    addBandwidthTypedMetadata(metadata, sizeIt->second, gpuActivity.end - gpuActivity.begin);
    return metadata;
  }

  // if compute kernel, add grid and block
  auto gridIt = correlationToGrid.find(gpuActivity.id);
  auto blockIt = correlationToBlock.find(gpuActivity.id);
  if (gridIt != correlationToGrid.end() && blockIt != correlationToBlock.end()) {
    metadata.set(RocmMetadataFields::kGrid, gridIt->second);
    metadata.set(RocmMetadataFields::kBlock, blockIt->second);
  }

  return metadata;
}

// Runtime Activities

template <class T>
inline bool RuntimeActivity<T>::flowStart() const {
  bool should_correlate = raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc || raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipFree ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch;
  return should_correlate;
}

template <class T>
inline void RuntimeActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

template <>
inline const std::string RuntimeActivity<rocprofKernelRow>::metadataJson() const {
  std::string kernel = "";
  if ((raw().functionAddr != nullptr)) {
    kernel = fmt::format(
        R"JSON(
    "kernel": "{}", )JSON",
        demangle(hipKernelNameRefByPtr(raw().functionAddr, raw().stream)));
  } else if ((raw().function != nullptr)) {
    kernel = fmt::format(
        R"JSON(
    "kernel": "{}", )JSON",
        demangle(hipKernelNameRef(raw().function)));
  }
  // cache grid and block so we can pass it into async activity (GPU track)
  correlationToGrid[raw().id] = rocprofKernelGrid(raw());
  correlationToBlock[raw().id] = rocprofKernelBlock(raw());

  return fmt::format(
      R"JSON(
      {}"cid": {}, "correlation": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "shared memory": {})JSON",
      kernel,
      raw().cid,
      raw().id,
      raw().gridX,
      raw().gridY,
      raw().gridZ,
      raw().workgroupX,
      raw().workgroupY,
      raw().workgroupZ,
      raw().groupSegmentSize);
}

template <>
inline TypedMetadata RuntimeActivity<rocprofKernelRow>::typedMetadata() const {
  TypedMetadata metadata;
  if ((raw().functionAddr != nullptr)) {
    metadata.set(RocmMetadataFields::kKernel, demangle(hipKernelNameRefByPtr(raw().functionAddr, raw().stream)));
  } else if ((raw().function != nullptr)) {
    metadata.set(RocmMetadataFields::kKernel, demangle(hipKernelNameRef(raw().function)));
  }

  // cache grid and block so we can pass it into async activity (GPU track)
  correlationToGrid[raw().id] = rocprofKernelGrid(raw());
  correlationToBlock[raw().id] = rocprofKernelBlock(raw());

  metadata.set(RocmMetadataFields::kCid, static_cast<int64_t>(raw().cid));
  metadata.set(RocmMetadataFields::kCorrelation, static_cast<int64_t>(raw().id));
  metadata.set(RocmMetadataFields::kGrid, correlationToGrid[raw().id]);
  metadata.set(RocmMetadataFields::kBlock, correlationToBlock[raw().id]);
  metadata.set(RocmMetadataFields::kSharedMemory, static_cast<int64_t>(raw().groupSegmentSize));
  return metadata;
}

template <>
inline const std::string RuntimeActivity<rocprofCopyRow>::metadataJson() const {
  correlationToSize[raw().id] = raw().size;
  return fmt::format(
      R"JSON(
      "cid": {}, "correlation": {}, "src": "{}", "dst": "{}", "bytes": "{}", "kind": "{}")JSON",
      raw().cid,
      raw().id,
      raw().src,
      raw().dst,
      raw().size,
      fmt::underlying(raw().kind));
}

template <>
inline TypedMetadata RuntimeActivity<rocprofCopyRow>::typedMetadata() const {
  TypedMetadata metadata;
  correlationToSize[raw().id] = raw().size;
  metadata.set(RocmMetadataFields::kCid, static_cast<int64_t>(raw().cid));
  metadata.set(RocmMetadataFields::kCorrelation, static_cast<int64_t>(raw().id));
  metadata.set(RocmMetadataFields::kSrc, fmt::format("{}", raw().src));
  metadata.set(RocmMetadataFields::kDst, fmt::format("{}", raw().dst));
  metadata.set(RocmMetadataFields::kBytes, static_cast<int64_t>(raw().size));
  metadata.set(RocmMetadataFields::kKind, fmt::format("{}", fmt::underlying(raw().kind)));
  return metadata;
}

template <>
inline const std::string RuntimeActivity<rocprofMallocRow>::metadataJson() const {
  correlationToSize[raw().id] = raw().size;
  std::string size = "";
  if (raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc) {
    size = fmt::format(
        R"JSON(
      "bytes": {}, )JSON",
        raw().size);
  }
  return fmt::format(
      R"JSON(
      {}"cid": {}, "correlation": {}, "ptr": "{}")JSON",
      size,
      raw().cid,
      raw().id,
      raw().ptr);
}

template <>
inline TypedMetadata RuntimeActivity<rocprofMallocRow>::typedMetadata() const {
  TypedMetadata metadata;
  correlationToSize[raw().id] = raw().size;
  if (raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc) {
    metadata.set(RocmMetadataFields::kBytes, static_cast<int64_t>(raw().size));
  }
  metadata.set(RocmMetadataFields::kCid, static_cast<int64_t>(raw().cid));
  metadata.set(RocmMetadataFields::kCorrelation, static_cast<int64_t>(raw().id));
  metadata.set(RocmMetadataFields::kPtr, fmt::format("{}", raw().ptr));
  return metadata;
}

template <class T>
inline const std::string RuntimeActivity<T>::metadataJson() const {
  return fmt::format(
      R"JSON(
      "cid": {}, "correlation": {})JSON",
      raw().cid,
      raw().id);
}

template <class T>
inline TypedMetadata RuntimeActivity<T>::typedMetadata() const {
  TypedMetadata metadata;
  metadata.set(RocmMetadataFields::kCid, static_cast<int64_t>(raw().cid));
  metadata.set(RocmMetadataFields::kCorrelation, static_cast<int64_t>(raw().id));
  return metadata;
}

} // namespace KINETO_NAMESPACE
