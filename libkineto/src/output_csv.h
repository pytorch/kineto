/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "SampleListener.h"

#include <fstream>
#include <ostream>
#include <set>

namespace KINETO_NAMESPACE {

class EventCSVLogger : public SampleListener {
 public:
  void update(const Config& config) override;
  void handleSample(int device, const Sample& sample, bool from_new_version) override;

 protected:
  EventCSVLogger() : out_(nullptr) {}

  std::ostream* out_;
  std::set<std::string> eventNames_;
  std::vector<int> percentiles_;
};

class EventCSVFileLogger : public EventCSVLogger {
 public:
  void update(const Config& config) override;

 private:
  std::ofstream of_;
  std::string filename_;
};

class EventCSVDbgLogger : public EventCSVLogger {
 public:
  void update(const Config& config) override;
};

} // namespace KINETO_NAMESPACE
