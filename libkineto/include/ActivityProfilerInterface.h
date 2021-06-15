/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "ActivityType.h"
#include "ActivityTraceInterface.h"

namespace libkineto {

class ActivityProfilerController;
struct CpuTraceBuffer;
class Config;

class ActivityProfilerInterface {

 public:
  virtual ~ActivityProfilerInterface() {};

  virtual void init() {}
  virtual bool isInitialized() {
    return false;
  }
  virtual bool isActive(){
    return false;
  }

  // *** Asynchronous API ***
  // Instead of starting and stopping the trace manually, provide a start time
  // and duration and / or iteration stop criterion.
  // Tracing terminates when either condition is met.
  virtual void scheduleTrace(const std::string& configStr) {}

  // *** Synchronous API ***
  // These must be called in order:
  // prepareTrace -> startTrace -> stopTrace.

  // Many tracing structures are lazily initialized during trace collection,
  // with potentially high overhead.
  // Call prepareTrace to enable tracing, then run the region to trace
  // at least once (and ideally run the same code that is to be traced) to
  // allow tracing structures to be initialized.
  // TODO: Add optional config string param
  virtual void prepareTrace(const std::set<ActivityType>& activityTypes) {}

  // Start recording, potentially reusing any buffers allocated since
  // prepareTrace was called.
  virtual void startTrace() {}

  // Stop and process trace, producing an in-memory list of trace records.
  // The processing will be done synchronously (using the calling thread.)
  virtual std::unique_ptr<ActivityTraceInterface> stopTrace() {
    return nullptr;
  }

  // *** TraceActivity API ***
  // FIXME: Pass activityProfiler interface into clientInterface?
  virtual void pushCorrelationId(uint64_t id){}
  virtual void popCorrelationId(){}
  virtual void transferCpuTrace(
      std::unique_ptr<CpuTraceBuffer> traceBuffer){}

  // Correlation ids for user defined spans
  virtual void pushUserCorrelationId(uint64_t){}
  virtual void popUserCorrelationId(){}

  // Include regions with this name
  virtual bool enableForRegion(const std::string& match) {
    return true;
  }

  // Saves information for the current thread to be used in profiler output
  // Client must record any new kernel thread where the activity has occured.
  virtual void recordThreadInfo() {}

  // Record trace metadata, currently supporting only string key and values,
  // values with the same key are overwritten
  virtual void addMetadata(const std::string& key, const std::string& value) = 0;
};

} // namespace libkineto
