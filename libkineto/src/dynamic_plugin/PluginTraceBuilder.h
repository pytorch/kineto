#pragma once

#include "ActivityType.h"
#include "GenericTraceActivity.h"
#include "Logger.h"
#include "libkineto.h"

#include "KinetoDynamicPluginInterface.h"
#include "PluginUtils.h"

namespace libkineto {

// Event builder provides a simple abstraction for plugins to interace with the
// event system in kineto. This works across binaries and has to be dealt with
// care. We do not use any C++ stdlib component across the interface and thus we
// need to translate a few things around

class PluginTraceBuilder {
 public:
  PluginTraceBuilder(TraceSpan span) {
    buffer_ = std::make_unique<CpuTraceBuffer>();
    buffer_->span = span;
  }

  // Several of these APIs do not have failure cases, but we still return an
  // integer for future extensibility.

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int addEvent(const KinetoPlugin_ProfileEvent* pProfileEvent) {
    if (buffer_ == nullptr) {
      return -1;
    }

    if (pProfileEvent == nullptr) {
      LOG(ERROR) << "Failed to add event of nullptr";
      return -1;
    }

    // Handle versioning
    // Currently expect the exact same version
    if (pProfileEvent->unpaddedStructSize <
        KINETO_PLUGIN_PROFILER_PROCESS_EVENTS_PARAMS_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Profile event has an incompatible version";
      return -1;
    }

    ActivityType activityType = convertToActivityType(pProfileEvent->eventType);

    buffer_->emplace_activity(
        buffer_->span, activityType /*ActivityType*/, "" /*name - set it later*/
    );

    auto& event = buffer_->activities.back();
    event->startTime = pProfileEvent->startTimeUtcNs;
    event->endTime = pProfileEvent->endTimeUtcNs;
    event->id = pProfileEvent->eventId;
    event->device = pProfileEvent->deviceId;
    event->resource = pProfileEvent->resourceId;
    event->threadId = pProfileEvent->threadId;

    return 0;
  }

  // returns 0 on success, -1 on failure, this can happen if the last event is not set.
  int setLastEventName(const char* pName) {
    if (buffer_ == nullptr) {
      return -1;
    }

    if (pName == nullptr) {
      LOG(ERROR) << "Failed to set last event name of nullptr";
      return -1;
    }

    if (buffer_->activities.empty()) {
      LOG(ERROR) << "Failed to set last event flow as there is no last event";
      return -1;
    }

    buffer_->activities.back()->activityName.assign(pName);
    return 0;
  }

  int setLastEventFlow(const KinetoPlugin_ProfileEventFlow* pProfileEventFlow) {
    if (buffer_ == nullptr) {
      return -1;
    }

    if (pProfileEventFlow == nullptr) {
      LOG(ERROR) << "Failed to set last event flow of nullptr";

      return -1;
    }

    // Handle versioning
    // Currently expect the exact same version
    if (pProfileEventFlow->unpaddedStructSize <
        KINETO_PLUGIN_PROFILE_EVENT_FLOW_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Profile event flow has an incompatible version";
      return -1;
    }

    if (buffer_->activities.empty()) {
      LOG(ERROR) << "Failed to set last event flow as there is no last event";
      return -1;
    }

    auto& event = buffer_->activities.back();

    event->flow.id = pProfileEventFlow->flowId;
    event->flow.type = convertToLinkType(pProfileEventFlow->flowType);
    event->flow.start = pProfileEventFlow->isStartPoint ? 1 : 0;

    return 0;
  }

  // returns 0 on success, -1 on failure, this can happen if the last event is not set.
  int addLastEventMetadata(const char* pKey, const char* pValue) {
    if (buffer_ == nullptr) {
      return -1;
    }

    if (pKey == nullptr || pValue == nullptr) {
      LOG(ERROR) << "Failed to set last event metadata of nullptr";
      return -1;
    }

    if (buffer_->activities.empty()) {
      LOG(ERROR)
          << "Failed to set last event metadata as there is no last event";
      return -1;
    }

    buffer_->activities.back()->addMetadata(
        std::string{pKey}, std::string{pValue});
    return 0;
  }

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int addDeviceInfo(const KinetoPlugin_ProfileDeviceInfo* pProfileDeviceInfo) {
    if (pProfileDeviceInfo == nullptr) {
      LOG(ERROR) << "Failed to add device info of nullptr";
      return -1;
    }

    // Handle versioning
    // Currently expect the exact same version
    if (pProfileDeviceInfo->unpaddedStructSize <
        KINETO_PLUGIN_PROFILE_DEVICE_INFO_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Profile device info has an incompatible version";
      return -1;
    }

    DeviceInfo deviceInfo(
        pProfileDeviceInfo->deviceId,
        pProfileDeviceInfo->sortIndex,
        pProfileDeviceInfo->pName
            ? std::string(pProfileDeviceInfo->pName)
            : std::to_string(pProfileDeviceInfo->deviceId),
        pProfileDeviceInfo->pLabel ? std::string(pProfileDeviceInfo->pLabel)
                                   : "");

    deviceInfos_.push_back(deviceInfo);

    return 0;
  }

  // returns 0 on success, -1 on failure, generally this is not expected to fail.
  int addResourceInfo(
      const KinetoPlugin_ProfileResourceInfo* pProfileResourceInfo) {
    if (pProfileResourceInfo == nullptr) {
      LOG(ERROR) << "Failed to add resource info of nullptr";
      return -1;
    }

    // Handle versioning
    // Currently expect the exact same version
    if (pProfileResourceInfo->unpaddedStructSize <
        KINETO_PLUGIN_PROFILE_RESOURCE_INFO_UNPADDED_STRUCT_SIZE) {
      LOG(ERROR) << "Profile resource info has an incompatible version";
      return -1;
    }

    ResourceInfo resourceInfo(
        pProfileResourceInfo->deviceId,
        pProfileResourceInfo->resourceId,
        pProfileResourceInfo->displayOrder,
        pProfileResourceInfo->pName
            ? std::string(pProfileResourceInfo->pName)
            : std::to_string(pProfileResourceInfo->resourceId));

    resourceInfos_.push_back(resourceInfo);

    return 0;
  }

  const KinetoPlugin_TraceBuilder toCTraceBuilder() {
    return KinetoPlugin_TraceBuilder{
        .unpaddedStructSize = KINETO_PLUGIN_TRACE_BUILDER_UNPADDED_STRUCT_SIZE,
        .pTraceBuilderHandle =
            reinterpret_cast<KinetoPlugin_TraceBuilderHandle*>(this),
        .addEvent = cAddEvent,
        .setLastEventName = cSetLastEventName,
        .setLastEventFlow = cSetLastEventFlow,
        .addLastEventMetadata = cAddLastEventMetadata,
        .addDeviceInfo = cAddDeviceInfo,
        .addResourceInfo = cAddResourceInfo};
  }

  // release ownership of the trace events and metadata
  std::unique_ptr<CpuTraceBuffer> getTraceBuffer() {
    return std::move(buffer_);
  }

  std::vector<DeviceInfo> getDeviceInfos() {
    return deviceInfos_;
  }

  std::vector<ResourceInfo> getResourceInfos() {
    return resourceInfos_;
  }

 private:
  static int cAddEvent(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const KinetoPlugin_ProfileEvent* pProfileEvent) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->addEvent(pProfileEvent);
  }

  static int cSetLastEventName(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const char* name) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->setLastEventName(name);
  }

  static int cSetLastEventFlow(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const KinetoPlugin_ProfileEventFlow* pProfileEventFlow) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->setLastEventFlow(pProfileEventFlow);
  }

  static int cAddLastEventMetadata(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const char* pKey,
      const char* pValue) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->addLastEventMetadata(pKey, pValue);
  }

  static int cAddDeviceInfo(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const KinetoPlugin_ProfileDeviceInfo* pProfileDeviceInfo) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->addDeviceInfo(pProfileDeviceInfo);
  }

  static int cAddResourceInfo(
      KinetoPlugin_TraceBuilderHandle* pTraceBuilderHandle,
      const KinetoPlugin_ProfileResourceInfo* pProfileResourceInfo) {
    auto pPluginTraceBuilder =
        reinterpret_cast<PluginTraceBuilder*>(pTraceBuilderHandle);
    return pPluginTraceBuilder->addResourceInfo(pProfileResourceInfo);
  }

  std::unique_ptr<CpuTraceBuffer> buffer_;
  std::vector<DeviceInfo> deviceInfos_;
  std::vector<ResourceInfo> resourceInfos_;
};

} // namespace libkineto
