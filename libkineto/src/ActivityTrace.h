// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string>

#include "ActivityLoggerFactory.h"
#include "ActivityTraceInterface.h"
#include "output_json.h"
#include "output_membuf.h"

namespace libkineto {

class ActivityTrace : public ActivityTraceInterface {
 public:
  ActivityTrace(
      std::unique_ptr<MemoryTraceLogger> tmpLogger,
      const ActivityLoggerFactory& factory,
      bool shouldPushToUST = false)
    : memLogger_(std::move(tmpLogger)),
      loggerFactory_(factory),
      pushToUST(shouldPushToUST) {
  }
  ~ActivityTrace() override;
  const std::vector<const ITraceActivity*>* activities() override;
  void save(const std::string& url) override;

 private:
  // Activities are logged into a buffer
  std::unique_ptr<MemoryTraceLogger> memLogger_;

  // Alternative logger used by save() if protocol prefix is specified
  const ActivityLoggerFactory& loggerFactory_;

  bool pushToUST;
};

} // namespace libkineto
