/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiProfilerMacros.h"

#include <fmt/format.h>

#include <stdexcept>

namespace KINETO_NAMESPACE {

[[noreturn]] void throwXpuRuntimeError(
    std::string_view errMsg,
    pti_result errCode) {
  auto errMsgWithCode =
      fmt::format("{} The error code is {}", errMsg, static_cast<int>(errCode));
#if PTI_VERSION_AT_LEAST(0, 10)
  errMsgWithCode = fmt::format(
      "{}. The detailed error message is: {}",
      errMsgWithCode,
      ptiResultTypeToString(errCode));
#endif
  throw std::runtime_error(errMsgWithCode);
}

[[noreturn]] void
throwXpuRuntimeError(const char* func, int line, pti_result errCode) {
  auto errMsg = fmt::format(
      "Kineto Profiler on XPU got error from function {} line {}.", func, line);
  throwXpuRuntimeError(errMsg, errCode);
}

} // namespace KINETO_NAMESPACE
