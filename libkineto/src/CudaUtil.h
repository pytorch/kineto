// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace KINETO_NAMESPACE {

#ifdef HAS_CUPTI
bool isGpuAvailable();
#else
bool isGpuAvailable() { return false; }
#endif

} // namespace KINETO_NAMESPACE
