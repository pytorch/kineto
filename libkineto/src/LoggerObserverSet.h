/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <set>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace KINETO_NAMESPACE {

class LoggerObserverSet {
 public:
  void insert(ILoggerObserver* observer) {
    std::lock_guard<std::mutex> guard(loggerObserversMutex_);
    loggerObservers_.insert(observer);
  }

  void remove(ILoggerObserver* observer) {
    std::lock_guard<std::mutex> guard(loggerObserversMutex_);
    loggerObservers_.erase(observer);
  }

  void write(const std::string message, LoggerOutputType ot) {
    std::lock_guard<std::mutex> guard(loggerObserversMutex_);
    // Output to observers. Current Severity helps keep track of which bucket the output goes.
    for (auto observer : loggerObservers_) {
      if (observer) {
        observer->write(message, ot);
      }
    }
  }

 private:
  std::set<ILoggerObserver*> loggerObservers_;
  std::mutex loggerObserversMutex_;
};

} // namespace KINETO_NAMESPACE
