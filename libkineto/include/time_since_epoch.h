// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>

namespace libkineto {

template <class ClockT>
inline int64_t timeSinceEpoch(
      const std::chrono::time_point<ClockT>& t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               t.time_since_epoch())
        .count();
}

} // namespace libkineto
