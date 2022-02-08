// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>

namespace libkineto {

inline int64_t timeSinceEpoch(
      const std::chrono::time_point<std::chrono::system_clock>& t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               t.time_since_epoch())
        .count();
}

} // namespace libkineto
