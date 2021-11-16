/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !USE_GOOGLE_LOG

#include <map>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace KINETO_NAMESPACE {

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    buckets_[ot].push_back(message);
  }

  const std::map<LoggerOutputType, std::vector<std::string>> extractCollectorMetadata() override {
    return buckets_;
  }

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
