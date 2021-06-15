/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#include <list>
#include <memory>

#include "libkineto.h"
#include "CuptiActivityBuffer.h"

namespace KINETO_NAMESPACE {

struct ActivityBuffers {
  std::list<std::unique_ptr<libkineto::CpuTraceBuffer>> cpu;
  std::unique_ptr<CuptiActivityBufferMap> gpu;
};

} // namespace KINETO_NAMESPACE
