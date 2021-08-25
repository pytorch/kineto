/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#ifdef HAS_CUPTI
#include <cupti.h>
#include "CuptiActivity.h"
#endif // HAS_CUPTI
#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

namespace KINETO_NAMESPACE {
  class Config;
  class GpuKernelActivity;
  struct RuntimeActivity;
}

namespace libkineto {

using namespace KINETO_NAMESPACE;

class ActivityLogger {
 public:

  virtual ~ActivityLogger() = default;

  struct DeviceInfo {
    DeviceInfo(int64_t id, const std::string& name, const std::string& label) :
      id(id), name(name), label(label) {}
    int64_t id;
    const std::string name;
    const std::string label;
  };

  struct ResourceInfo {
    ResourceInfo(int64_t deviceId, int64_t id, const std::string& name) :
        id(id), deviceId(deviceId), name(name) {}
    int64_t id;
    int64_t deviceId;
    const std::string name;
  };

  virtual void handleDeviceInfo(
      const DeviceInfo& info,
      uint64_t time) = 0;

  virtual void handleResourceInfo(const ResourceInfo& info, int64_t time) = 0;

  virtual void handleTraceSpan(const TraceSpan& span) = 0;

  virtual void handleGenericActivity(
      const libkineto::GenericTraceActivity& activity) = 0;

#ifdef HAS_CUPTI
  virtual void handleRuntimeActivity(const RuntimeActivity& activity) = 0;

  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityKernel4>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemcpy>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemcpy2>& activity) = 0;
  virtual void handleGpuActivity(
      const GpuActivity<CUpti_ActivityMemset>& activity) = 0;
#endif // HAS_CUPTI

  virtual void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) = 0;

  void handleTraceStart() {
    handleTraceStart(std::unordered_map<std::string, std::string>());
  }

  virtual void finalizeTrace(
      const KINETO_NAMESPACE::Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) = 0;

 protected:
  ActivityLogger() = default;
};

} // namespace KINETO_NAMESPACE
