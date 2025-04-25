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

#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

namespace {
thread_local std::unordered_map<int, std::string> correlationToGrid;
thread_local std::unordered_map<int, std::string> correlationToBlock;
thread_local std::unordered_map<int, size_t> correlationToSize;
} // namespace

const char* getGpuActivityKindString(uint32_t domain, uint32_t op) {
  if (domain == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH)
    return "Dispatch Kernel";
  else if (domain == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
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

void getMemcpySrcDstString(uint32_t kind, std::string& src, std::string& dst) {
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
    std::string opString = RocprofLogger::opString(
        static_cast<rocprofiler_callback_tracing_kind_t>(domain), op);
    const char* name = opString.c_str();
    return demangle(
        raw().kernelName.length() > 0 ? raw().kernelName : std::string(name));
  } else if (type_ == ActivityType::GPU_MEMSET) {
    return fmt::format(
        "Memset ({})", getGpuActivityKindString(raw().domain, raw().op));
  } else if (type_ == ActivityType::GPU_MEMCPY) {
    std::string src = "";
    std::string dst = "";
    getMemcpySrcDstString(raw().op, src, dst);
    return fmt::format(
        "Memcpy {} ({} -> {})",
        getGpuActivityKindString(raw().domain, raw().op),
        src,
        dst);
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

inline const std::string GpuActivity::metadataJson() const {
  const auto& gpuActivity = raw();
  // clang-format off

  // if memcpy or memset, add size
  if (correlationToSize.count(gpuActivity.id) > 0) {
    size_t size = correlationToSize[gpuActivity.id];
    std::string bandwidth_gib = (bandwidth(size, gpuActivity.end - gpuActivity.begin));
    return fmt::format(R"JSON(
      "device": {}, "stream": {},
      "correlation": {}, "kind": "{}",
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      gpuActivity.device, gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op),
      size, bandwidth_gib);
  } 
  
  // if compute kernel, add grid and block
  else if (correlationToGrid.count(gpuActivity.id) > 0) {
    return fmt::format(R"JSON(
      "device": {}, "stream": {},
      "correlation": {}, "kind": "{}",
      "grid": {}, "block": {})JSON",
      gpuActivity.device, gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op),
      correlationToGrid[gpuActivity.id], correlationToBlock[gpuActivity.id]);
  } else {
    return fmt::format(R"JSON(
      "device": {}, "stream": {},
      "correlation": {}, "kind": "{}")JSON",
      gpuActivity.device, gpuActivity.queue,
      gpuActivity.id, getGpuActivityKindString(gpuActivity.domain, gpuActivity.op));
  }
  // clang-format on
}

// Runtime Activities

template <class T>
inline bool RuntimeActivity<T>::flowStart() const {
  bool should_correlate =
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipFree ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream;
  return should_correlate;
}

template <class T>
inline void RuntimeActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

template <>
inline const std::string RuntimeActivity<rocprofKernelRow>::metadataJson()
    const {
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
  correlationToGrid[raw().id] = fmt::format(
      R"JSON(
    [{}, {}, {}])JSON",
      raw().gridX,
      raw().gridY,
      raw().gridZ);

  correlationToBlock[raw().id] = fmt::format(
      R"JSON(
    [{}, {}, {}])JSON",
      raw().workgroupX,
      raw().workgroupY,
      raw().workgroupZ);

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
inline const std::string RuntimeActivity<rocprofMallocRow>::metadataJson()
    const {
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

template <class T>
inline const std::string RuntimeActivity<T>::metadataJson() const {
  return fmt::format(
      R"JSON(
      "cid": {}, "correlation": {})JSON",
      raw().cid,
      raw().id);
}

} // namespace KINETO_NAMESPACE
