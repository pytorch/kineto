/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>

namespace libkineto {
/* [Note: Temp Libkineto Nanosecond]
This is a temporary hack to support nanosecond time units in Libkineto.
After pytorch changes are made to support nanosecond precision, this
can be removed.
*/
template <class ClockT>
inline int64_t timeSinceEpoch(
      const std::chrono::time_point<ClockT>& t) {
#ifdef TMP_LIBKINETO_NANOSECOND
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
#endif
               t.time_since_epoch())
        .count();
}

} // namespace libkineto
