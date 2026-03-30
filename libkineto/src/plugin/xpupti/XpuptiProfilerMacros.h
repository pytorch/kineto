/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string_view>

#include <pti/pti.h>

namespace KINETO_NAMESPACE {

using namespace libkineto;

#define PTI_VERSION_AT_LEAST(MAJOR, MINOR) \
  (PTI_VERSION_MAJOR > MAJOR || (PTI_VERSION_MAJOR == MAJOR && PTI_VERSION_MINOR >= MINOR))

[[noreturn]] void throwXpuRuntimeError(std::string_view errMsg, pti_result errCode);

[[noreturn]] void throwXpuRuntimeError(const char* func, int line, pti_result errCode);

#define XPUPTI_CALL(returnCode)                             \
  {                                                         \
    if (returnCode != PTI_SUCCESS) {                        \
      throwXpuRuntimeError(__func__, __LINE__, returnCode); \
    }                                                       \
  }

using DeviceIndex_t = int8_t;

} // namespace KINETO_NAMESPACE
