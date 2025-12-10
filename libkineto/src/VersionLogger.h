/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace KINETO_NAMESPACE {

class DeviceVersionLogger {
 public:
  DeviceVersionLogger(std::recursive_mutex& mutex) : mutex_(mutex) {};
  virtual ~DeviceVersionLogger() = default;
  virtual void logAndRecordVersions() = 0;
  std::unordered_map<std::string, std::string> getVersionMetadata() {
    return versionMetadata_;
  }

 protected:
  void addVersionMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    versionMetadata_[key] = value;
  }

 private:
  std::unordered_map<std::string, std::string> versionMetadata_;
  std::recursive_mutex& mutex_;
};

class CudaVersionLogger : public DeviceVersionLogger {
 public:
  CudaVersionLogger(std::recursive_mutex& mutex) : DeviceVersionLogger(mutex) {}
  void logAndRecordVersions() override;
};

class HipVersionLogger : public DeviceVersionLogger {
 public:
  HipVersionLogger(std::recursive_mutex& mutex) : DeviceVersionLogger(mutex) {}
  void logAndRecordVersions() override;
};

std::unique_ptr<DeviceVersionLogger> selectDeviceVersionLogger(
    std::recursive_mutex& mutex);

} // namespace KINETO_NAMESPACE
