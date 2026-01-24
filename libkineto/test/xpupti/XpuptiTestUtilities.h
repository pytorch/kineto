/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/output_base.h"

namespace KN = KINETO_NAMESPACE;

bool IsEnvVerbose();

std::pair<
    std::unique_ptr<KN::IActivityProfilerSession>,
    std::unique_ptr<KN::CpuTraceBuffer>>
RunProfilerTest(
    const std::vector<std::string_view>& metrics,
    const std::set<KN::ActivityType>& activities,
    const KN::Config& cfg,
    unsigned repeatCount,
    const std::vector<std::string>& expectedActivities,
    const std::vector<std::string>& expectedTypes);
