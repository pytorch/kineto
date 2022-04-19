// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/format.h>

#ifdef HAS_CUPTI

#include <cuda.h>

#define CUDA_CALL(call)                                      \
  [&]() -> cudaError_t {                                     \
    cudaError_t _status_ = call;                             \
    if (_status_ != cudaSuccess) {                           \
      const char* _errstr_ = cudaGetErrorString(_status_);   \
      LOG(WARNING) << fmt::format(                           \
          "function {} failed with error {} ({})",           \
          #call,                                             \
          _errstr_,                                          \
          (int)_status_);                                    \
    }                                                        \
    return _status_;                                         \
  }()

#endif // HAS_CUPTI
