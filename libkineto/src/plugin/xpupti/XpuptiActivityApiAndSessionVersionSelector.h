/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

#if PTI_VERSION_AT_LEAST(0, 15)

class XpuptiActivityApiV2;

#define XPUPTI_ACTIVITY_API XpuptiActivityApiV2
#define XPUPTI_ACTIVITY_PROFILER_SESSION XpuptiActivityProfilerSessionV2

#else

class XpuptiActivityApi;

#define XPUPTI_ACTIVITY_API XpuptiActivityApi
#define XPUPTI_ACTIVITY_PROFILER_SESSION XpuptiActivityProfilerSession

#endif

} // namespace KINETO_NAMESPACE
