/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
