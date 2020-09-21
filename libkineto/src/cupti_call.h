/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>

// TODO: Handle lazy loading error differently (invalid _status_)
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
