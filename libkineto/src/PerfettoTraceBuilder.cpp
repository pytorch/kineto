/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PerfettoTraceBuilder.h"

#include <fstream>
#include <random>
#include "ITraceActivity.h"
#include "Logger.h"
#include "TraceSpan.h"
#include "output_base.h"
#include "parfait/protos/perfetto/trace/perfetto_trace.pb.h"

namespace KINETO_NAMESPACE {

const std::string kTrackNamePlaceHolder = "PLACE_HOLDER";

PerfettoTraceBuilder::PerfettoTraceBuilder()
    : trace_(std::make_unique<Trace>()) {
  // Generate random packet sequence ID
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
  packetSequenceId_ = dis(gen);

  // Create the main process descriptor
  processUuid_ = nextUuid_++;
  createProcessDescriptor("Kineto Trace", processUuid_);
}

PerfettoTraceBuilder::~PerfettoTraceBuilder() = default;

TrackDescriptor* PerfettoTraceBuilder::createProcessDescriptor(
    const std::string& processName,
    uint64_t uuid) {
  auto* packet = trace_->add_packet();
  packet->set_trusted_packet_sequence_id(packetSequenceId_);

  auto* track_descriptor = packet->mutable_track_descriptor();
  track_descriptor->set_uuid(uuid);
  track_descriptor->set_name(processName);

  auto* process = track_descriptor->mutable_process();
  process->set_pid(0); // Default PID
  process->set_process_name(processName);

  return track_descriptor;
}

TrackDescriptor* PerfettoTraceBuilder::createThreadDescriptor(
    const std::string& threadName,
    uint64_t uuid,
    uint64_t parentUuid,
    int32_t pid,
    int32_t tid) {
  auto* packet = trace_->add_packet();
  packet->set_trusted_packet_sequence_id(packetSequenceId_);

  auto* track_descriptor = packet->mutable_track_descriptor();
  track_descriptor->set_uuid(uuid);
  track_descriptor->set_parent_uuid(parentUuid);
  track_descriptor->set_name(threadName);

  auto* thread = track_descriptor->mutable_thread();
  thread->set_pid(pid);
  thread->set_tid(tid);
  thread->set_thread_name(threadName);

  return track_descriptor;
}

TrackDescriptor* PerfettoTraceBuilder::getOrCreateDeviceProcess(
    int64_t deviceId) {
  auto it = deviceProcessDescriptors_.find(deviceId);
  if (it != deviceProcessDescriptors_.end()) {
    return it->second;
  }

  // Create new process descriptor for this device with placeholder name
  std::string placeholderName = "Device " + std::to_string(deviceId);
  TrackDescriptor* process_descriptor =
      createProcessDescriptor(placeholderName, nextUuid_++);

  deviceProcessDescriptors_[deviceId] = process_descriptor;

  return process_descriptor;
}

TrackDescriptor* PerfettoTraceBuilder::createChildTrack(
    TrackDescriptor* parentTrack) {
  uint64_t subTrackUuid = nextUuid_++;

  auto* packet = trace_->add_packet();
  TrackDescriptor* subTrack = packet->mutable_track_descriptor();

  subTrack->set_uuid(subTrackUuid);
  subTrack->set_parent_uuid(parentTrack->uuid());
  subTrack->set_name(parentTrack->name());

  subTrackMap[parentTrack].emplace_back(subTrack);

  return subTrack;
}

TrackDescriptor* PerfettoTraceBuilder::getOrCreateTrack(
    int64_t deviceId,
    int64_t resourceId,
    int32_t threadId) {
  std::string key = std::to_string(deviceId) + "_" + std::to_string(resourceId);
  auto it = trackUuids_.find(key);
  if (it != trackUuids_.end()) {
    return it->second;
  }

  // Get or create the device process first
  TrackDescriptor* deviceProcess = getOrCreateDeviceProcess(deviceId);

  // Create new track
  TrackDescriptor* track_descriptor = createThreadDescriptor(
      kTrackNamePlaceHolder,
      nextUuid_++,
      deviceProcess->uuid(),
      static_cast<int32_t>(deviceId),
      threadId != 0 ? threadId : static_cast<int32_t>(resourceId));

  trackUuids_[key] = track_descriptor;
  return track_descriptor;
}

void PerfettoTraceBuilder::handleDeviceInfo(const libkineto::DeviceInfo& info) {
  // Check if we already created a process descriptor for this device
  auto it = deviceProcessDescriptors_.find(info.id);
  if (it != deviceProcessDescriptors_.end()) {
    // Update the existing process descriptor name
    it->second->set_name(info.name + " [" + info.label + "]");
    it->second->mutable_process()->set_pid(static_cast<int32_t>(info.id));
    it->second->mutable_process()->set_process_name(info.name);
  } else {
    // Create a new process descriptor for the device
    getOrCreateDeviceProcess(info.id);
    auto* descriptor = deviceProcessDescriptors_[info.id];

    // Update with real device info
    descriptor->set_name(info.name + " [" + info.label + "]");
    descriptor->mutable_process()->set_pid(static_cast<int32_t>(info.id));
    descriptor->mutable_process()->set_process_name(info.name);
  }
}

void PerfettoTraceBuilder::handleOverheadInfo(
    const ActivityLogger::OverheadInfo& info,
    int64_t time) {
  // Create an instant event for overhead info
  auto* packet = trace_->add_packet();
  packet->set_trusted_packet_sequence_id(packetSequenceId_);
  packet->set_timestamp(time);

  auto* track_event = packet->mutable_track_event();
  track_event->set_type(TrackEvent::TYPE_INSTANT);
  track_event->set_name("Overhead: " + info.name);
  track_event->set_track_uuid(processUuid_);
}

void PerfettoTraceBuilder::handleResourceInfo(
    const libkineto::ResourceInfo& info) {
  // Check if this resource already has a track created
  std::string resourceKey =
      std::to_string(info.deviceId) + "_" + std::to_string(info.id);

  auto it = trackUuids_.find(resourceKey);
  if (it != trackUuids_.end()) {
    // Track already exists, update its name
    it->second->mutable_thread()->set_thread_name(info.name);
    it->second->set_name(info.name);
  } else {
    // Create new track descriptor for this resource
    uint64_t resourceUuid = nextUuid_++;
    auto* track_descriptor = createThreadDescriptor(
        info.name,
        resourceUuid,
        processUuid_,
        static_cast<int32_t>(info.deviceId),
        static_cast<int32_t>(info.id));

    // Store the pointer for future updates
    trackUuids_[resourceKey] = track_descriptor;
  }
}

void PerfettoTraceBuilder::handleTraceSpan(const TraceSpan& span) {
  // Create begin event
  auto* begin_packet = trace_->add_packet();
  begin_packet->set_trusted_packet_sequence_id(packetSequenceId_);
  begin_packet->set_timestamp(span.startTime);

  auto* begin_event = begin_packet->mutable_track_event();
  begin_event->set_type(TrackEvent::TYPE_SLICE_BEGIN);
  begin_event->set_name(span.name);
  begin_event->set_track_uuid(processUuid_);

  // Create end event
  auto* end_packet = trace_->add_packet();
  end_packet->set_trusted_packet_sequence_id(packetSequenceId_);
  end_packet->set_timestamp(span.endTime);

  auto* end_event = end_packet->mutable_track_event();
  end_event->set_type(TrackEvent::TYPE_SLICE_END);
  end_event->set_track_uuid(processUuid_);
}

void PerfettoTraceBuilder::handleActivity(const ITraceActivity& activity) {
  // Get or create track for this activity
  std::string trackName = activity.name();
  TrackDescriptor* track = getOrCreateTrack(
      activity.deviceId(), activity.resourceId(), activity.getThreadId());

  TrackDescriptor* subTrackUuid = createChildTrack(track);

  // Create begin event
  auto* begin_packet = trace_->add_packet();
  begin_packet->set_trusted_packet_sequence_id(packetSequenceId_);
  begin_packet->set_timestamp(activity.timestamp());

  auto* begin_event = begin_packet->mutable_track_event();
  begin_event->set_type(TrackEvent::TYPE_SLICE_BEGIN);
  begin_event->set_name(activity.name());
  begin_event->set_track_uuid(subTrackUuid->uuid());

  // Add metadata as debug annotations if available
  std::string metadataJson = activity.metadataJson();
  if (!metadataJson.empty()) {
    auto* debug_annotation = begin_event->add_debug_annotations();
    debug_annotation->set_name("metadata");
    debug_annotation->set_string_value(metadataJson);
  }

  // Add correlation ID as debug annotation
  if (activity.correlationId() != 0) {
    auto* corr_annotation = begin_event->add_debug_annotations();
    corr_annotation->set_name("correlation_id");
    corr_annotation->set_int_value(activity.correlationId());
  }

  // Add flow IDs if present
  if (activity.flowId() != 0) {
    begin_event->add_flow_ids(activity.flowId());
  }

  // Create end event
  int64_t endTime = activity.timestamp() + activity.duration();
  auto* end_packet = trace_->add_packet();
  end_packet->set_trusted_packet_sequence_id(packetSequenceId_);
  end_packet->set_timestamp(endTime);

  auto* end_event = end_packet->mutable_track_event();
  end_event->set_type(TrackEvent::TYPE_SLICE_END);
  end_event->set_track_uuid(subTrackUuid->uuid());
}

void PerfettoTraceBuilder::handleGenericActivity(
    const GenericTraceActivity& activity) {
  handleActivity(activity);
}

void PerfettoTraceBuilder::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {
  // Add metadata as instant events with debug annotations
  for (const auto& [key, value] : metadata) {
    auto* packet = trace_->add_packet();
    packet->set_trusted_packet_sequence_id(packetSequenceId_);

    auto* track_event = packet->mutable_track_event();
    track_event->set_type(TrackEvent::TYPE_INSTANT);
    track_event->set_name("trace_metadata");
    track_event->set_track_uuid(processUuid_);

    auto* annotation = track_event->add_debug_annotations();
    annotation->set_name(key);
    annotation->set_string_value(value);
  }

  // Add device properties
  if (!device_properties.empty()) {
    auto* packet = trace_->add_packet();
    packet->set_trusted_packet_sequence_id(packetSequenceId_);

    auto* track_event = packet->mutable_track_event();
    track_event->set_type(TrackEvent::TYPE_INSTANT);
    track_event->set_name("device_properties");
    track_event->set_track_uuid(processUuid_);

    auto* annotation = track_event->add_debug_annotations();
    annotation->set_name("properties");
    annotation->set_string_value(device_properties);
  }
}

bool PerfettoTraceBuilder::writeToFile(const std::string& filename) {
  std::ofstream output(filename, std::ios::out | std::ios::binary);

  for (const auto& pair : subTrackMap) {
    const TrackDescriptor* parent = pair.first;
    const std::vector<TrackDescriptor*>& childTracks = pair.second;
    for (TrackDescriptor* childTrack : childTracks) {
      childTrack->set_name(parent->name());
    }
  }

  if (!output) {
    LOG(ERROR) << "Failed to open Perfetto protobuf trace file: " << filename;
    return false;
  }

  if (!trace_->SerializeToOstream(&output)) {
    LOG(ERROR) << "Failed to serialize trace to Perfetto protobuf file: "
               << filename;
    output.close();
    return false;
  }

  output.close();
  LOG(INFO) << "Perfetto protobuf trace written to: " << filename;
  return true;
}

} // namespace KINETO_NAMESPACE
