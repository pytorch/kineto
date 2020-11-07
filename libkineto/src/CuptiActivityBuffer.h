/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>
#include <sys/types.h>
#include <unistd.h>

#include "TraceActivity.h"
#include "cupti_strings.h"

namespace KINETO_NAMESPACE {

class CuptiActivityBuffer {
 public:
  // data must be allocated using malloc.
  // Ownership is transferred to this object.
  CuptiActivityBuffer(uint8_t* data, size_t validSize)
      : data(data), validSize(validSize) {}

  ~CuptiActivityBuffer() {
    free(data);
  }

  // Allocated by malloc
  uint8_t* data{nullptr};

  // Number of bytes used
  size_t validSize;
};

} // namespace KINETO_NAMESPACE
