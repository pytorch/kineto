/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Config.h"

#include <chrono>
#include <set>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

constexpr char kCuptiProfilerConfigName[] = "cupti_rb_profiler";

class CuptiRangeProfilerConfig : public AbstractConfig {
 public:
  bool handleOption(const std::string& name, std::string& val) override;

  void validate(
      const std::chrono::time_point<std::chrono::system_clock>&
      fallbackProfileStartTime) override {}

  static CuptiRangeProfilerConfig& get(const Config& cfg) {
    return dynamic_cast<CuptiRangeProfilerConfig&>(cfg.feature(
          kCuptiProfilerConfigName));
  }

  Config& parent() const {
    return *parent_;
  }

  std::vector<std::string> activitiesCuptiMetrics() const {
    return activitiesCuptiMetrics_;
  }

  bool cuptiProfilerPerKernel() const {
    return cuptiProfilerPerKernel_;
  }

  int64_t cuptiProfilerMaxRanges() const {
    return cuptiProfilerMaxRanges_;
  }

  void setSignalDefaults() override {
    setDefaults();
  }

  void setClientDefaults() override {
    setDefaults();
  }

  void printActivityProfilerConfig(std::ostream& s) const override;
  void setActivityDependentConfig() override;
  static void registerFactory();
 protected:
  AbstractConfig* cloneDerived(AbstractConfig& parent) const override {
    CuptiRangeProfilerConfig* clone = new CuptiRangeProfilerConfig(*this);
    clone->parent_ = dynamic_cast<Config*>(&parent);
    return clone;
  }

 private:
 CuptiRangeProfilerConfig() = delete;
  explicit CuptiRangeProfilerConfig(Config& parent);
  explicit CuptiRangeProfilerConfig(
      const CuptiRangeProfilerConfig& other) = default;

  // some defaults will depend on other configuration
  void setDefaults();

  // Associated Config object
  Config* parent_;

  // Counter metrics exposed via CUPTI Profiler API
  std::vector<std::string> activitiesCuptiMetrics_;

  // Collect profiler metrics per kernel - autorange made
  bool cuptiProfilerPerKernel_{false};

  // max number of ranges to configure the profiler for.
  // this has to be set before hand to reserve space for the output
  int64_t cuptiProfilerMaxRanges_ = 0;
};

} // namespace KINETO_NAMESPACE
