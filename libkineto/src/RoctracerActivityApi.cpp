/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <chrono>
#include <functional>
#include <time.h>

#include "RoctracerActivityApi.h"
#include "RoctracerLogger.h"
#include "Demangle.h"
#include "output_base.h"
#include "ThreadUtil.h"

typedef uint64_t timestamp_t;

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

using namespace std::chrono;

// Local copy of hip op types.  These are public (and stable) in later rocm releases
typedef enum {
  HIP_OP_COPY_KIND_UNKNOWN_ = 0,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_ = 0x11F3,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_ = 0x11F4,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_ = 0x11F5,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_ = 0x1201,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_ = 0x1202,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_ = 0x1203,
  HIP_OP_COPY_KIND_FILL_BUFFER_ = 0x1207
} hip_op_copy_kind_t_;

typedef enum {
  HIP_OP_DISPATCH_KIND_UNKNOWN_ = 0,
  HIP_OP_DISPATCH_KIND_KERNEL_ = 0x11F0,
  HIP_OP_DISPATCH_KIND_TASK_ = 0x11F1
} hip_op_dispatch_kind_t_;

typedef enum {
  HIP_OP_BARRIER_KIND_UNKNOWN_ = 0
} hip_op_barrier_kind_t_;
// end hip op defines

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(2 * 1024 * 1024);

RoctracerActivityApi& RoctracerActivityApi::singleton() {
  static RoctracerActivityApi instance;
  return instance;
}

RoctracerActivityApi::RoctracerActivityApi()
: d(&RoctracerLogger::singleton()) {
}

RoctracerActivityApi::~RoctracerActivityApi() {
  disableActivities(std::set<ActivityType>());
}

void RoctracerActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->pushCorrelationID(id, static_cast<RoctracerLogger::CorrelationDomain>(type));
#endif
}

void RoctracerActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->popCorrelationID(static_cast<RoctracerLogger::CorrelationDomain>(type));
#endif
}

void RoctracerActivityApi::setMaxBufferSize(int size) {
  // FIXME: implement?
  //maxGpuBufferCount_ = 1 + size / kBufSize;
}

inline bool inRange(int64_t start, int64_t end, int64_t stamp) {
  return ((stamp > start) && (stamp < end));
}

inline bool RoctracerActivityApi::isLogged(libkineto::ActivityType atype) {
  return activityMaskSnapshot_ & (1 << static_cast<uint32_t>(atype));
}

int RoctracerActivityApi::processActivities(
    ActivityLogger& logger, std::function<const ITraceActivity*(int32_t)> linkedActivity,
    int64_t startTime, int64_t endTime) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  timespec t0, t1, t00;
  clock_gettime(CLOCK_REALTIME, &t0);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  clock_gettime(CLOCK_REALTIME, &t00);

  const timestamp_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);
  // Our stored timestamps (from roctracer and generated) are in CLOCK_MONOTONIC domain (in ns).
  // Convert the guards rather than each timestamp.
  startTime = (startTime * 1000) - toffset;
  endTime = (endTime * 1000) - toffset;

  int count = 0;

  auto &externalCorrelations = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Domain0];

  // Basic Api calls

  if (isLogged(ActivityType::CUDA_RUNTIME)) {

    for (auto &item : d->rows_) {
      if (!inRange(startTime, endTime, item.begin))
          continue;
      GenericTraceActivity a;
      a.startTime = (item.begin + toffset) / 1000;
      a.endTime = (item.end + toffset) / 1000;
      a.id = item.id;
      a.device = item.pid;
      a.resource = item.tid;
      a.activityType = ActivityType::CUDA_RUNTIME;
      a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));
      a.flow.id = item.id;
      a.flow.type = kLinkAsyncCpuGpu;
      a.flow.start = true;

      auto it = externalCorrelations.find(a.id);
      a.linked = linkedActivity(it == externalCorrelations.end() ? 0 : it->second);

      logger.handleGenericActivity(a);
      ++count;
    }

    // Malloc/Free calls
    for (auto &item : d->mallocRows_) {
      if (!inRange(startTime, endTime, item.begin))
          continue;
      GenericTraceActivity a;
      a.startTime = (item.begin + toffset) / 1000;
      a.endTime = (item.end + toffset) / 1000;
      a.id = item.id;
      a.device = item.pid;
      a.resource = item.tid;
      a.activityType = ActivityType::CUDA_RUNTIME;
      a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));
      a.flow.id = item.id;
      a.flow.type = kLinkAsyncCpuGpu;
      a.flow.start = true;

      auto it = externalCorrelations.find(a.id);
      a.linked = linkedActivity(it == externalCorrelations.end() ? 0 : it->second);

      a.addMetadataQuoted("ptr", fmt::format("{}", item.ptr));
      if (item.cid == HIP_API_ID_hipMalloc) {
        a.addMetadata("size", item.size);
      }

      logger.handleGenericActivity(a);
      ++count;
    }

    // HipMemcpy calls
    for (auto &item : d->copyRows_) {
      if (!inRange(startTime, endTime, item.begin))
          continue;
      GenericTraceActivity a;
      a.startTime = (item.begin + toffset) / 1000;
      a.endTime = (item.end + toffset) / 1000;
      a.id = item.id;
      a.device = item.pid;
      a.resource = item.tid;
      a.activityType = ActivityType::CUDA_RUNTIME;
      a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));
      a.flow.id = item.id;
      a.flow.type = kLinkAsyncCpuGpu;
      a.flow.start = true;

      auto it = externalCorrelations.find(a.id);
      a.linked = linkedActivity(it == externalCorrelations.end() ? 0 : it->second);

      a.addMetadataQuoted("src", fmt::format("{}", item.src));
      a.addMetadataQuoted("dst", fmt::format("{}", item.dst));
      a.addMetadata("size", item.size);
      a.addMetadata("kind", fmt::format("{}", fmt::underlying(item.kind)));
      if ((item.cid == HIP_API_ID_hipMemcpyAsync) || (item.cid == HIP_API_ID_hipMemcpyWithStream)) {
        a.addMetadataQuoted("stream", fmt::format("{}", reinterpret_cast<void*>(item.stream)));
      }

      logger.handleGenericActivity(a);
      ++count;
    }

    // Kernel Launch Api calls

    for (auto &item : d->kernelRows_) {
      if (!inRange(startTime, endTime, item.begin))
          continue;
      GenericTraceActivity a;
      a.startTime = (item.begin + toffset) / 1000;
      a.endTime = (item.end + toffset) / 1000;
      a.id = item.id;
      a.device = item.pid;
      a.resource = item.tid;
      a.activityType = ActivityType::CUDA_RUNTIME;
      a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));
      a.flow.id = item.id;
      a.flow.type = kLinkAsyncCpuGpu;
      a.flow.start = true;

      auto it = externalCorrelations.find(a.id);
      a.linked = linkedActivity(it == externalCorrelations.end() ? 0 : it->second);

      // TODO: Use lowercase kernel, once legacy tools update.
      if (item.functionAddr != nullptr) {
        a.addMetadataQuoted(
            "Kernel", demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)));
      }
      else if (item.function != nullptr) {
        a.addMetadataQuoted(
            "Kernel", demangle(hipKernelNameRef(item.function)));
      }
      a.addMetadata("grid dim", fmt::format("[{}, {}, {}]", item.gridX, item.gridY, item.gridZ));
      a.addMetadata("block dim", fmt::format("[{}, {}, {}]", item.workgroupX, item.workgroupY, item.workgroupZ));
      a.addMetadata("shared size", item.groupSegmentSize);
      a.addMetadataQuoted("stream", fmt::format("{}", reinterpret_cast<void*>(item.stream)));

      logger.handleGenericActivity(a);
      ++count;
    }
  }  // isLogged(ActivityType::CUDA_RUNTIME)

  // Async Ops

  for (auto &item : d->opRows_) {
    if (!inRange(startTime, endTime, item.begin))
          continue;
    GenericTraceActivity a;

    // Overlay launch metadata for kernels
    auto kit = kernelLaunches_.find(item.id);
    if (kit != kernelLaunches_.end()) {
      a = (*kit).second;
    }

    const char *name = roctracer_op_string(item.domain, item.op, item.kind);
    a.device = item.device;
    a.resource = item.queue;

    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;

    a.activityType = ActivityType::CONCURRENT_KERNEL;
    a.activityName = item.kernelName.length() > 0 ? item.kernelName : std::string(name);
    a.flow.id = item.id;
    a.flow.type = kLinkAsyncCpuGpu;
    a.flow.start = false;

    auto eit = externalCorrelations.find(a.id);
    a.linked = linkedActivity(eit == externalCorrelations.end() ? 0 : eit->second);

    bool filtered = false;

    switch (item.kind) {
      case HIP_OP_COPY_KIND_DEVICE_TO_HOST_:
      case HIP_OP_COPY_KIND_HOST_TO_DEVICE_:
      case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_:
      case HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_:
      case HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_:
      case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_:
        if (!isLogged(ActivityType::GPU_MEMCPY))
          filtered = true;
        a.activityType = ActivityType::GPU_MEMCPY;
        break;
      case HIP_OP_COPY_KIND_FILL_BUFFER_:
        if (!isLogged(ActivityType::GPU_MEMSET))
          filtered = true;
        a.activityType = ActivityType::GPU_MEMSET;
        break;
      case HIP_OP_DISPATCH_KIND_KERNEL_:
      case HIP_OP_DISPATCH_KIND_TASK_:
      default:
        if (!isLogged(ActivityType::CONCURRENT_KERNEL))
          filtered = true;
        if (item.op == HIP_OP_ID_BARRIER)  // Don't record barriers/markers
          filtered = true;
        a.activityType = ActivityType::CONCURRENT_KERNEL;
        break;
    }

    if (!filtered && inRange(startTime, endTime, item.begin)) {
        logger.handleGenericActivity(a);
        ++count;
    }
  }
  return count;
}

void RoctracerActivityApi::clearActivities() {
  d->clearLogs();
  kernelLaunches_.clear();
  kernelNames_.clear();    
}


void RoctracerActivityApi::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  d->startLogging();

  for (const auto& activity : selected_activities) {
    activityMask_ |= (1 << static_cast<uint32_t>(activity));
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        d->externalCorrelationEnabled_ = true;
    }
  }
#endif
}

void RoctracerActivityApi::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  d->stopLogging();

  activityMaskSnapshot_ = activityMask_;

  for (const auto& activity : selected_activities) {
    activityMask_ &= ~(1 << static_cast<uint32_t>(activity));
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        d->externalCorrelationEnabled_ = false;
    }
  }
#endif
}

} // namespace KINETO_NAMESPACE
