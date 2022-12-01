/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <functional>

namespace libkineto {

template <class ClockT>
inline int64_t timeSinceEpoch(
      const std::chrono::time_point<std::chrono::system_clock>& t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      t.time_since_epoch()).count();
}

int64_t timeSinceEpoch();

#ifdef KINETO_UNIT_TEST
void setMockTimeSinceEpoch(std::function<int64_t()> f);
#endif

inline int64_t toMilliseconds(int64_t us) {
  return us / 1000;
}

} // namespace libkineto
