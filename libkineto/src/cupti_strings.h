/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

namespace libkineto {

const char* memoryKindString(CUpti_ActivityMemoryKind kind);
const char* memcpyKindString(CUpti_ActivityMemcpyKind kind);
const char* runtimeCbidName(CUpti_CallbackId cbid);
const char* overheadKindString(CUpti_ActivityOverheadKind kind);
const char* syncTypeString(CUpti_ActivitySynchronizationType kind);

} // namespace libkineto
