/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace KINETO_NAMESPACE {

// cupti's timestamps are platform specific. This function convert the raw
// cupti timestamp to time since unix epoch. So that on different platform,
// correction can work correctly.
uint64_t unixEpochTimestamp(uint64_t ts);

} // namespace KINETO_NAMESPACE
