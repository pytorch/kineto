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

namespace KINETO_NAMESPACE {

class Config;

class ActivityLogger {
 public:
  virtual ~ActivityLogger() = default;

  // Handle any configuration before staring the trace
  virtual void configure(const Config& config) = 0;

  virtual void handleProcessName(
      pid_t pid,
      const std::string& processName,
      const std::string& label,
      uint64_t time) = 0;

  virtual void
  handleThreadName(uint32_t tid, const std::string& label, uint64_t time) = 0;

  virtual void handleNetCPUSpan(
      int netId,
      const std::string& netName,
      int iteration,
      int opCount,
      int gpuOpCount,
      uint64_t startTime,
      uint64_t stopTime) = 0;

  virtual void handleNetGPUSpan(
      int netId,
      const std::string& netName,
      int iteration,
      uint64_t startTime,
      uint64_t stopTime) = 0;

  virtual void handleIterationStart(
      const std::string& netName,
      int64_t time,
      uint32_t tid) = 0;

  virtual void handleCpuActivity(
      const std::string& netName,
      int netIteration,
      const libkineto::external_api::OpDetails& activity) = 0;

  virtual void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity,
      const libkineto::external_api::OpDetails& ext) = 0;

  virtual void handleGpuActivity(
      const CUpti_ActivityKernel4* kernel,
      const libkineto::external_api::OpDetails& ext,
      int smCount) = 0;

  virtual void handleGpuActivity(
      const CUpti_ActivityMemcpy* memcpy,
      const libkineto::external_api::OpDetails& ext) = 0;

  virtual void handleGpuActivity(
      const CUpti_ActivityMemcpy2* memcpy,
      const libkineto::external_api::OpDetails& ext) = 0;

  virtual void handleGpuActivity(
      const CUpti_ActivityMemset* memset,
      const libkineto::external_api::OpDetails& ext) = 0;

  // Create a flow event to an external event
  virtual void handleLinkStart(const CUpti_ActivityAPI* activity) = 0;
  virtual void
  handleLinkEnd(uint32_t id, int device, int stream, uint64_t tsUsecs) = 0;

  virtual void finalizeTrace(const Config& config) = 0;

 protected:
  ActivityLogger() = default;

  // get a cleaner thread id
  int renameThreadID(size_t tid) {
    // the tid here is the thread ID that schedules the operator
    static int curr_tid = 0;

    // Note this function is not thread safe; The user of this
    // ActivityLogger need to maintain thread safety
    if (tidMap_.count(tid)) {
      return tidMap_[tid];
    } else {
      return tidMap_[tid] = curr_tid++;
    }
  }

 private:
  // store the mapping of thread id vs. showing on the trace
  std::unordered_map<size_t, int> tidMap_;
};

} // namespace KINETO_NAMESPACE
