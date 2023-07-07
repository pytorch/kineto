/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  void GenericTraceActivity::log(ActivityLogger& logger) const {
    logger.handleGenericActivity(*this);
  }
} // namespace libkineto
