/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_csv.h"

#include <chrono>
#include <fstream>
#include <iomanip>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "Config.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {

static void write_header(
    std::ostream& out,
    const std::vector<int>& percentiles) {
  out << "timestamp,delta_ms,device,event_name";
  for (int p : percentiles) {
    out << ",p" << p;
  }
  out << ",total" << std::endl;
}

void EventCSVLogger::update(const Config& config) {
  eventNames_.clear();
  eventNames_.insert(config.eventNames().begin(), config.eventNames().end());
  eventNames_.insert(config.metricNames().begin(), config.metricNames().end());
  if (config.percentiles() != percentiles_) {
    percentiles_ = config.percentiles();
    if (out_) {
      write_header(*out_, percentiles_);
    }
  }
}

void EventCSVLogger::handleSample(int device, const Sample& sample, bool from_new_version) {
  using namespace std::chrono;
  if (out_) {
    auto now = system_clock::now();
    auto time = system_clock::to_time_t(now);
    for (const Stat& s : sample.stats) {
      if (eventNames_.find(s.name) == eventNames_.end()) {
        continue;
      }
      *out_ << fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(time)) << ",";
      *out_ << sample.deltaMsec << ",";
      *out_ << device << ",";
      *out_ << s.name;
      for (const auto& p : s.percentileValues) {
        *out_ << "," << p.second;
      }
      *out_ << "," << s.total << std::endl;
    }
  }
}

void EventCSVFileLogger::update(const Config& config) {
  if (config.eventLogFile() != filename_) {
    if (of_.is_open()) {
      of_.close();
      out_ = nullptr;
      percentiles_.clear();
    }
    filename_ = config.eventLogFile();
    if (!filename_.empty()) {
      of_.open(filename_, std::ios::out | std::ios::trunc);
      out_ = &of_;
    }
  }
  EventCSVLogger::update(config);
}

void EventCSVDbgLogger::update(const Config& config) {
  if (out_ && config.verboseLogLevel() < 0) {
    out_ = nullptr;
  } else if (!out_ && config.verboseLogLevel() >= 0) {
    out_ = &LIBKINETO_DBG_STREAM;
  }
  if (config.verboseLogLevel() >= 0) {
    percentiles_.clear();
    EventCSVLogger::update(config);
  }
}

} // namespace KINETO_NAMESPACE
