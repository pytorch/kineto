// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  void GenericTraceActivity::log(ActivityLogger& logger) const {
    logger.handleGenericActivity(*this);
  }
} // namespace libkineto
