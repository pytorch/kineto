/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#if !USE_GOOGLE_LOG
#include <memory>
#endif // !USE_GOOGLE_LOG

namespace KINETO_NAMESPACE {

class IDaemonConfigLoader {
 public:
  virtual ~IDaemonConfigLoader() {}

  // Return the base config from the daemon
  virtual std::string readBaseConfig() = 0;

  // Return a configuration string from the daemon, if one has been posted.
  virtual std::string readOnDemandConfig(bool events, bool activities) = 0;

  // Returns the number of tracked contexts for this device. The daemon has a
  // global view. If an unexpedted error occurs, return -1.
  virtual int gpuContextCount(uint32_t device) = 0;

  virtual void setCommunicationFabric(bool enabled) = 0;
};

// Basic Daemon Config Loader that uses IPCFabric for communication
// Only works on Linux based platforms
#ifdef __linux__
class DaemonConfigLoader : public IDaemonConfigLoader {
 public:
  DaemonConfigLoader() {}

  // Return the base config from the daemon
  std::string readBaseConfig() override;

  // Return a configuration string from the daemon, if one has been posted.
  std::string readOnDemandConfig(bool events, bool activities) override;

  // Returns the number of tracked contexts for this device. The daemon has a
  // global view. If an unexpected error occurs, return -1.
  int gpuContextCount(uint32_t device) override;

  void setCommunicationFabric(bool enabled) override;

  static void registerFactory();
};
#endif // __linux__


} // namespace KINETO_NAMESPACE
