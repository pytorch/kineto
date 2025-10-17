/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <string>

#include <pti/pti_version.h>

namespace KINETO_NAMESPACE {

using namespace libkineto;

#define PTI_VERSION_AT_LEAST(MAJOR, MINOR) \
  (PTI_VERSION_MAJOR > MAJOR ||            \
   (PTI_VERSION_MAJOR == MAJOR && PTI_VERSION_MINOR >= MINOR))

#if PTI_VERSION_AT_LEAST(0, 10)
#define XPUPTI_CALL(returnCode)                                                \
  {                                                                            \
    if (returnCode != PTI_SUCCESS) {                                           \
      std::string funcMsg(__func__);                                           \
      std::string codeMsg = std::to_string(returnCode);                        \
      std::string HeadMsg("Kineto Profiler on XPU got error from function ");  \
      std::string Msg(". The error code is ");                                 \
      std::string detailMsg(". The detailed error message is ");               \
      detailMsg = detailMsg + std::string(ptiResultTypeToString(returnCode));  \
      throw std::runtime_error(HeadMsg + funcMsg + Msg + codeMsg + detailMsg); \
    }                                                                          \
  }
#else
#define XPUPTI_CALL(returnCode)                                               \
  {                                                                           \
    if (returnCode != PTI_SUCCESS) {                                          \
      std::string funcMsg(__func__);                                          \
      std::string codeMsg = std::to_string(returnCode);                       \
      std::string HeadMsg("Kineto Profiler on XPU got error from function "); \
      std::string Msg(". The error code is ");                                \
      throw std::runtime_error(HeadMsg + funcMsg + Msg + codeMsg);            \
    }                                                                         \
  }
#endif

class XpuptiActivityApi;
using DeviceIndex_t = int8_t;

} // namespace KINETO_NAMESPACE
