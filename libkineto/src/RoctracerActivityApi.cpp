/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RoctracerActivityApi.h"

#include "ApproximateClock.h"
#include <cstring>
#include <chrono>
#include <functional>
#include <time.h>
#include "Logger.h"
#include "Demangle.h"
#include "output_base.h"
#include "ThreadUtil.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

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

void RoctracerActivityApi::setTimeOffset(timestamp_t toffset) {
  toffset_ = toffset;
}

int RoctracerActivityApi::processActivities(
    std::function<void(const roctracerBase*)> handler,
    std::function<void(uint64_t, uint64_t, RoctracerLogger::CorrelationDomain)> correlationHandler) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  int count = 0;

  // Process all external correlations pairs
  for (int it = RoctracerLogger::CorrelationDomain::begin; it < RoctracerLogger::CorrelationDomain::end; ++it) {
    auto &externalCorrelations = d->externalCorrelations_[it];
    for (auto &item : externalCorrelations) {
      correlationHandler(item.first, item.second, static_cast<RoctracerLogger::CorrelationDomain>(it));
    }
    std::lock_guard<std::mutex> lock(d->externalCorrelationsMutex_);
    externalCorrelations.clear();
  }

  // All Runtime API Calls
  for (auto &item : d->rows_) {
    bool filtered = false;
    if (item->type != ROCTRACER_ACTIVITY_ASYNC && !isLogged(ActivityType::CUDA_RUNTIME)) {
      filtered = true;
    }
    else {
      switch (reinterpret_cast<roctracerAsyncRow*>(item)->kind) {
        case HIP_OP_COPY_KIND_DEVICE_TO_HOST_:
        case HIP_OP_COPY_KIND_HOST_TO_DEVICE_:
        case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_:
        case HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_:
        case HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_:
        case HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_:
          if (!isLogged(ActivityType::GPU_MEMCPY))
            filtered = true;
          break;
        case HIP_OP_COPY_KIND_FILL_BUFFER_:
          if (!isLogged(ActivityType::GPU_MEMSET))
            filtered = true;
          break;
        case HIP_OP_DISPATCH_KIND_KERNEL_:
        case HIP_OP_DISPATCH_KIND_TASK_:
        default:
          if (!isLogged(ActivityType::CONCURRENT_KERNEL))
            filtered = true;
          // Don't record barriers/markers
          if (reinterpret_cast<roctracerAsyncRow*>(item)->op == HIP_OP_ID_BARRIER)
            filtered = true;
          break;
      }
    }
    if (!filtered) {
      // Convert the begin and end timestamps from monotonic clock to system clock.
      item->begin = item->begin + toffset_;
      item->end = item->end + toffset_;
      handler(item);
      ++count;
    }
  }
  return count;
}

void RoctracerActivityApi::clearActivities() {
  d->clearLogs();
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
