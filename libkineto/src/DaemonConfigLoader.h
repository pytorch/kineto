// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <string>

namespace KINETO_NAMESPACE {

class DaemonConfigLoader {
 public:
  virtual ~DaemonConfigLoader() {}

  // Return the base config from the daemon
  virtual std::string readBaseConfig() = 0;

  // Return a configuration string from the daemon, if one has been posted.
  virtual std::string readOnDemandConfig(bool events, bool activities) = 0;

  // Returns the number of tracked contexts for this device. The daemon has a
  // global view. If an unexpedted error occurs, return -1.
  virtual int gpuContextCount(uint32_t device) = 0;

  virtual void setCommunicationFabric(bool enabled) = 0;
};

} // namespace KINETO_NAMESPACE
