 /*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivity.h"

#include <fmt/format.h>

#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

template<>
inline const std::string GpuActivity<CUpti_ActivityKernel4>::name() const {
  return demangle(raw().name);
}

template<>
inline ActivityType GpuActivity<CUpti_ActivityKernel4>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

static inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
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

template<>
inline ActivityType GpuActivity<CUpti_ActivityMemcpy2>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template<>
inline const std::string GpuActivity<CUpti_ActivityMemcpy2>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
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

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleRuntimeActivity(*this);
}

template<class T>
inline void GpuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleGpuActivity(*this);
}

} // namespace KINETO_NAMESPACE
