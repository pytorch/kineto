/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace KINETO_NAMESPACE {

enum class ActivityType {
    MEMCPY,
    MEMSET,
    CONCURRENT_KERNEL,
    EXTERNAL_CORRELATION,
    RUNTIME
};

} // namespace KINETO_NAMESPACE
