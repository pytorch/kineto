/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include "ActivityTraceInterface.h"
#include "CuptiActivityInterface.h"
#include "output_json.h"
#include "output_membuf.h"

namespace libkineto {

class ActivityTrace : public ActivityTraceInterface {
 public:
  ActivityTrace(
      std::unique_ptr<MemoryTraceLogger> logger,
      CuptiActivityInterface& cuptiActivities)
    : logger_(std::move(logger)), cuptiActivities_(cuptiActivities) {}

  const std::vector<std::unique_ptr<TraceActivity>>* activities() override {
    return logger_->traceActivities();
  };

  void save(const std::string& path) override {
    ChromeTraceLogger chrome_logger(path, cuptiActivities_.smCount());
    logger_->log(chrome_logger);
  };

 private:
  std::unique_ptr<MemoryTraceLogger> logger_;
  CuptiActivityInterface& cuptiActivities_;
};

} // namespace libkineto
