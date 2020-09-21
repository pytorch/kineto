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

#include <cupti.h>
#include "external_api.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

class Config;

class ChromeTraceLogger : public ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Prepare / refresh file
  void configure(const Config& config) override;

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleProcessName(
      pid_t pid,
      const std::string& processName,
      const std::string& label,
      uint64_t time) override;

  void handleThreadName(uint32_t tid, const std::string& label, uint64_t time)
      override;

  void handleNetCPUSpan(
      int netId,
      const std::string& netName,
      int iteration,
      int opCount,
      int gpuOpCount,
      uint64_t startTime,
      uint64_t stopTime) override;

  void handleNetGPUSpan(
      int netId,
      const std::string& netName,
      int iteration,
      uint64_t startTime,
      uint64_t stopTime) override;

  void handleIterationStart(
      const std::string& netName,
      int64_t time,
      uint32_t tid) override;

  void handleCpuActivity(
      const std::string& netName,
      int netIteration,
      const libkineto::external_api::OpDetails& activity) override;

  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity,
      const libkineto::external_api::OpDetails& ext) override;

  void handleGpuActivity(
      const CUpti_ActivityKernel4* kernel,
      const libkineto::external_api::OpDetails& ext,
      int smCount) override;

  void handleGpuActivity(
      const CUpti_ActivityMemcpy* memcpy,
      const libkineto::external_api::OpDetails& ext) override;

  void handleGpuActivity(
      const CUpti_ActivityMemcpy2* memcpy,
      const libkineto::external_api::OpDetails& ext) override;

  void handleGpuActivity(
      const CUpti_ActivityMemset* memset,
      const libkineto::external_api::OpDetails& ext) override;

  // Create a flow event to an external event
  void handleLinkStart(const CUpti_ActivityAPI* activity) override;
  void handleLinkEnd(uint32_t id, int device, int stream, uint64_t tsUsecs)
      override;

  void finalizeTrace(const Config& config) override;

 private:
  void logActivity(const CUpti_Activity* act);

  std::string fileName_;
  std::ofstream traceOf_;

  // store the mapping of thread id vs. showing on the trace
  std::unordered_map<uint32_t, int> tidMap_;

  // get a cleaner thread id
  int renameThreadID(uint32_t tid);

  // Cache pid to avoid repeated calls to getpid()
  pid_t pid_;
};

} // namespace KINETO_NAMESPACE
