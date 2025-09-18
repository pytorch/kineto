#pragma once

#include <libkineto.h>
#include <output_base.h>
#include <time_since_epoch.h>

#include <aiupti_activity.h>

namespace KINETO_NAMESPACE {

using namespace libkineto;

#define AIUPTI_CALL(returnCode)                                               \
  {                                                                           \
    if (returnCode != AIUPTI_SUCCESS) {                                       \
      std::string funcMsg(__func__);                                          \
      std::string codeMsg = std::to_string(returnCode);                       \
      std::string HeadMsg("Kineto Profiler on AIU got error from function "); \
      std::string Msg(". The error code is ");                                \
      throw std::runtime_error(HeadMsg + funcMsg + Msg + codeMsg);            \
    }                                                                         \
  }

class AiuptiActivityApi;
using DeviceIndex_t = int8_t;

} // namespace KINETO_NAMESPACE
