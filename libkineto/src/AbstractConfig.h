/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

class AbstractConfig {
 public:
  AbstractConfig& operator=(const AbstractConfig&) = delete;
  AbstractConfig(AbstractConfig&&) = delete;
  AbstractConfig& operator=(AbstractConfig&&) = delete;

  virtual ~AbstractConfig() {
    for (const auto& p : featureConfigs_) {
      delete p.second;
    }
  }

  // Return a copy of the full derived class
  virtual AbstractConfig* cloneDerived() const = 0;

  // Returns true if successfully parsed the config string
  bool parse(const std::string& conf);

  // Default setup for signal-triggered profiling
  virtual void setSignalDefaults() {
    for (auto& p : featureConfigs_) {
      p.second->setSignalDefaults();
    }
  }

  // Default setup for client-triggered profiling
  virtual void setClientDefaults() {
    for (auto& p : featureConfigs_) {
      p.second->setClientDefaults();
    }
  }

  // Time config was created / updated
  std::chrono::time_point<std::chrono::system_clock> timestamp() const {
    return timestamp_;
  }

  const std::string& source() const {
    return source_;
  }

  AbstractConfig& feature(std::string name) const {
    const auto& pos = featureConfigs_.find(name);
    return *pos->second;
  }

 protected:
  AbstractConfig() {}
  AbstractConfig(const AbstractConfig& other) = default;

  // Return true if the option was recognized and successfully parsed.
  // Throw std::invalid_argument if val is invalid.
  virtual bool handleOption(const std::string& name, std::string& val);

  // Perform post-validation checks, typically conditons involving
  // multiple options.
  // Throw std::invalid_argument if automatic correction can not be made.
  virtual void validate() = 0;

  // TODO: Separate out each profiler type into features?
  virtual void printActivityProfilerConfig(std::ostream& s) const;

  // Transfers ownership of cfg arg
  void addFeature(const std::string& name, AbstractConfig* cfg) {
    featureConfigs_[name] = cfg;
  }

  // Helpers for use in handleOption
  // Split a string by delimiter and remove external white space
  std::vector<std::string> splitAndTrim(const std::string& s, char delim) const;
  // Lowercase for case-insensitive comparisons
  std::string toLower(std::string& s) const;
  // Does string end with suffix
  bool endsWith(const std::string& s, const std::string& suffix) const;
  // Conversions
  int64_t toIntRange(const std::string& val, int64_t min, int64_t max) const;
  int32_t toInt32(const std::string& val) const;
  int64_t toInt64(const std::string& val) const;
  bool toBool(std::string& val) const;

  void cloneFeaturesInto(AbstractConfig& cfg) const {
    for (const auto& feature : featureConfigs_) {
      cfg.featureConfigs_[feature.first] = feature.second->cloneDerived();
    }
  }

 private:
  // Time config was created / updated
  std::chrono::time_point<std::chrono::system_clock> timestamp_{};

  // Original configuration string, used for comparison
  std::string source_{""};

  // Configuration objects for optional features
  std::map<std::string, AbstractConfig*> featureConfigs_{};
};

} // namespace KINETO_NAMESPACE
