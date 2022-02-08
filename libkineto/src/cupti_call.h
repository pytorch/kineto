// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/format.h>

#ifdef HAS_CUPTI

#include <cupti.h>

#define CUPTI_CALL(call)                           \
  [&]() -> CUptiResult {                           \
    CUptiResult _status_ = call;                   \
    if (_status_ != CUPTI_SUCCESS) {               \
      const char* _errstr_ = nullptr;              \
      cuptiGetResultString(_status_, &_errstr_);   \
      LOG(WARNING) << fmt::format(                 \
          "function {} failed with error {} ({})", \
          #call,                                   \
          _errstr_,                                \
          (int)_status_);                          \
    }                                              \
    return _status_;                               \
  }()

#define CUPTI_CALL_NOWARN(call) call

#else

#define CUPTI_CALL(call) call
#define CUPTI_CALL_NOWARN(call) call

#endif // HAS_CUPTI
