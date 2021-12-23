/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !USE_GOOGLE_LOG

#include <map>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

enum LoggerOutputType {
  VERBOSE = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  ENUM_COUNT = 4
};

const char* toString(LoggerOutputType t);
LoggerOutputType toLoggerOutputType(const std::string& str);

constexpr int LoggerTypeCount = (int) LoggerOutputType::ENUM_COUNT;

constexpr char kWarmUpStage[] = "Warm Up";
constexpr char kCollectionStage[] = "Collection";
constexpr char kPostProcessingStage[] = "Post Processing";

class ILoggerObserver {
 public:
  virtual ~ILoggerObserver() = default;
  virtual void write(const std::string& message, LoggerOutputType ot) = 0;
  virtual const std::map<LoggerOutputType, std::vector<std::string>> extractCollectorMetadata() = 0;

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
