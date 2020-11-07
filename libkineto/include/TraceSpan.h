/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <thread>

namespace libkineto {

struct TraceSpan {
  // FIXME: change to duration?
  int64_t startTime{0};
  int64_t endTime{0};
  int opCount{0};
  int iteration{-1};
  // Name is used to identify timeline
  std::string name;
  // Prefix used to distinguish sub-nets on the same timeline
  std::string prefix;
};

} // namespace libkineto
