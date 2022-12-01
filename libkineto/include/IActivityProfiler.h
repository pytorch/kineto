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
#include <set>
#include <vector>

#include "CpuTraceBuffer.h"
#include "GenericTraceActivity.h"
#include "IProfilerSession.h"

/* This file includes an abstract base class for an activity profiler
 * that can be implemented by multiple tracing agents in the application.
 * The high level Kineto profiler can co-ordinate start and end of tracing
 * and combine together events from multiple such activity profilers.
 */

namespace libkineto {

using namespace KINETO_NAMESPACE;
struct CpuTraceBuffer;

#ifdef _MSC_VER
// workaround for the predefined ERROR macro on Windows
#undef ERROR
#endif // _MSC_VER

class Config;
class ICompositeProfiler;
class ICompositeProfilerSession;

enum class TraceStatus {
  READY, // Success
  WARMUP, // Performing trace warmup
  RECORDING, // Actively collecting activities
  PROCESSING, // Recording is complete, preparing results
  ERROR, // One or more errors (and possibly also warnings) occurred.
  WARNING, // One or more warnings occurred.
};

/* DeviceInfo:
 *   Can be used to specify process name, PID and device label
 */
struct DeviceInfo {
  DeviceInfo(int64_t id, const std::string& name, const std::string& label)
      : id(id), name(name), label(label) {}
  int64_t id;               // process id
  const std::string name;   // process name
  const std::string label;  // device label
};

/* ResourceInfo:
 *   Can be used to specify resource inside device
 */
struct ResourceInfo {
  ResourceInfo(
      int64_t deviceId,
      int64_t id,
      int64_t sortIndex,
      const std::string& name)
      : id(id), sortIndex(sortIndex), deviceId(deviceId), name(name) {}
  int64_t id;             // resource id
  int64_t sortIndex;      // position in trace view
  int64_t deviceId;       // id of device which owns this resource (specified in DeviceInfo.id)
  const std::string name; // resource name
};
/* IActivityProfilerSession:
 *   an opaque object that can be used by a high level profiler to
 *   start/stop and return trace events.
 */
class IActivityProfilerSession : public IProfilerSession {

 public:
  virtual ~IActivityProfilerSession() {};

  virtual TraceStatus status() {
    return status_;
  }

  virtual void status(TraceStatus status) {
    if (status_ == TraceStatus::ERROR) {
      return;
    }
    if (status_ == TraceStatus::WARNING && status != TraceStatus::ERROR) {
      return;
    }
    status_ = status;
  }

  // processes trace activities using logger
  virtual void log(ActivityLogger& logger) = 0;

  // FIXME: Temporary hack until profilers are properly separated
  virtual void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {}

 protected:
  TraceStatus status_;
};


/* Activity Profiler Interface:
 *   These allow other frameworks to integrate into Kineto's primariy
 *   activity profiler. While the primary activity profiler handles
 *   timing the trace collections and correlating events the plugins
 *   can become source of new trace activity types.
 */
class IActivityProfiler {

 public:

  virtual ~IActivityProfiler() {};

  virtual const std::string name() const = 0;

  // returns activity types this profiler supports
  virtual const std::set<ActivityType>& supportedActivityTypes() const = 0;

  virtual void init(ICompositeProfiler* parent = nullptr) = 0;
  virtual bool isInitialized() const = 0;
  virtual bool isActive() const = 0;

  virtual std::shared_ptr<IActivityProfilerSession> configure(
      const Config& options,
      ICompositeProfilerSession* parentSession = nullptr) = 0;

  virtual void start(IActivityProfilerSession& session) = 0;
  virtual void stop(IActivityProfilerSession& session) = 0;
};

} // namespace libkineto
