/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mediator for initialization and profiler control

#pragma once

#include <deque>

#include "GenericTraceActivity.h"
#include "TraceSpan.h"

namespace libkineto {

struct CpuTraceBuffer {
  TraceSpan span{0, 0, "none"};
  int gpuOpCount;
  std::deque<GenericTraceActivity> activities;
};

} // namespace libkineto
