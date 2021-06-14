/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  void GenericTraceActivity::log(ActivityLogger& logger) const {
    // TODO(T89833634): Merge handleGenericTraceActivity and handleCpuActivity
    if (activityType == ActivityType::CPU_OP) {
      return;
    }

    logger.handleGenericActivity(*this);
  }
} // namespace libkineto
