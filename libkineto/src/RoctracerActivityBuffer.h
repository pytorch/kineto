/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>

namespace KINETO_NAMESPACE {

class RoctracerActivityBuffer {
 public:
  // data must be allocated using malloc.
  // Ownership is transferred to this object.
  RoctracerActivityBuffer(uint8_t* data, size_t validSize)
      : data(data), validSize(validSize) {}

  ~RoctracerActivityBuffer() {
    free(data);
  }

  // Allocated by malloc
  uint8_t* data{nullptr};

  // Number of bytes used
  size_t validSize;
};

} // namespace KINETO_NAMESPACE
