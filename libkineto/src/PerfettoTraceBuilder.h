/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <kineto/libkineto/src/perfetto_trace.pb.h>
#include <string>
#include <unordered_map>
#include "output_base.h"

namespace KINETO_NAMESPACE {

// PerfettoTraceBuilder is a helper class that constructs Perfetto protobuf
// traces It provides similar interface to ActivityLogger but builds a
// Perfetto trace object This class is designed to be used alongside
// ChromeTraceLogger to generate both JSON and Perfetto protobuf traces
class PerfettoTraceBuilder {
 public:
  PerfettoTraceBuilder();
  ~PerfettoTraceBuilder();

  // Handle trace events - similar to ActivityLogger interface
  void handleDeviceInfo(const libkineto::DeviceInfo& info);
  void handleOverheadInfo(
      const ActivityLogger::OverheadInfo& info,
      int64_t time);
  void handleResourceInfo(const libkineto::ResourceInfo& info);
  void handleTraceSpan(const TraceSpan& span);
  void handleActivity(const ITraceActivity& activity);
  void handleGenericActivity(const GenericTraceActivity& activity);
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties);

  // Write the Perfetto trace to file
  bool writeToFile(const std::string& filename);

  // Get the trace object (for testing or advanced usage)
  Trace* getTrace() const {
    return trace_.get();
  }

 private:
  // Get or create a track for a given device/resource/thread combination
  TrackDescriptor*
  getOrCreateTrack(int64_t deviceId, int64_t resourceId, int32_t threadId);
  TrackDescriptor* createChildTrack(TrackDescriptor* track);

  // Get or create process descriptor for a device
  TrackDescriptor* getOrCreateDeviceProcess(int64_t deviceId);

  // Create process descriptor packet and return pointer to it
  TrackDescriptor* createProcessDescriptor(
      const std::string& processName,
      uint64_t uuid);

  // Create thread descriptor packet and return pointer to it
  TrackDescriptor* createThreadDescriptor(
      const std::string& threadName,
      uint64_t uuid,
      uint64_t parentUuid,
      int32_t pid,
      int32_t tid);

  std::unique_ptr<Trace> trace_;

  // Track UUID management
  uint64_t nextUuid_{1};
  uint64_t processUuid_{0};

  // Map from (deviceId, resourceId, threadId) to track UUID
  std::unordered_map<std::string, TrackDescriptor*> trackUuids_;

  // Map from deviceId to process TrackDescriptor pointer for later name
  // updates
  std::unordered_map<int64_t, TrackDescriptor*> deviceProcessDescriptors_;

  // Map from Main Thread to child thread
  std::unordered_map<TrackDescriptor*, std::vector<TrackDescriptor*>>
      subTrackMap;

  // Packet sequence ID for trusted packets
  uint32_t packetSequenceId_{0};
};

} // namespace KINETO_NAMESPACE
