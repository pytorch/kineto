/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "time_since_epoch.h"

namespace libkineto {

static int64_t defaultTime() {
  return timeSinceEpoch(std::chrono::system_clock::now());
}

static std::function<int64_t()> timeFunc{defaultTime};
int64_t timeSinceEpoch() {
  return timeFunc();
}

void setMockTimeSinceEpoch(std::function<int64_t()> f) {
  timeFunc = f;
}

} // namespace libkineto
