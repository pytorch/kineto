#pragma once

#include <libkineto.h>
#include <output_base.h>
#include <time_since_epoch.h>

#include <pti/pti_view.h>
#include <sycl/sycl.hpp>

namespace KINETO_NAMESPACE {

using namespace libkineto;

#if PTI_VERSION_MAJOR > 0 || PTI_VERSION_MINOR > 9
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
