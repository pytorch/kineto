/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <VersionLogger.h>

namespace KINETO_NAMESPACE {

class XpuVersionLogger : public DeviceVersionLogger {
 public:
  XpuVersionLogger(std::recursive_mutex& mutex) : DeviceVersionLogger(mutex) {}
  void logAndRecordVersions() override;
};

} // namespace KINETO_NAMESPACE
