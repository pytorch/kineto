/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocprofActivityApi.h"

#include <time.h>
#include <chrono>
#include <cstring>
#include <functional>
#include "ApproximateClock.h"
#include "Demangle.h"
#include "Logger.h"
#include "ThreadUtil.h"
#include "output_base.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

RocprofActivityApi& RocprofActivityApi::singleton() {
  static RocprofActivityApi instance;
  return instance;
}

RocprofActivityApi::RocprofActivityApi() : d(&RocprofLogger::singleton()) {}

RocprofActivityApi::~RocprofActivityApi() {
  disableActivities(std::set<ActivityType>());
}

void RocprofActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->pushCorrelationID(
      id, static_cast<RocLogger::CorrelationDomain>(type));
#endif
}

void RocprofActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->popCorrelationID(
      static_cast<RocLogger::CorrelationDomain>(type));
#endif
}

void RocprofActivityApi::setMaxBufferSize(int size) {
  // FIXME: implement?
  // maxGpuBufferCount_ = 1 + size / kBufSize;
}

inline bool inRange(int64_t start, int64_t end, int64_t stamp) {
  return ((stamp > start) && (stamp < end));
}

inline bool RocprofActivityApi::isLogged(libkineto::ActivityType atype) const {
  return activityMaskSnapshot_ & (1 << static_cast<uint32_t>(atype));
}

void RocprofActivityApi::setTimeOffset(timestamp_t toffset) {
  toffset_ = toffset;
}

int RocprofActivityApi::processActivities(
    std::function<void(const rocprofBase*)> handler,
    std::function<void(uint64_t, uint64_t, RocLogger::CorrelationDomain)>
        correlationHandler) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  int count = 0;

  // Process all external correlations pairs
  for (int it = RocLogger::CorrelationDomain::begin;
       it < RocLogger::CorrelationDomain::end;
       ++it) {
    auto& externalCorrelations = d->externalCorrelations_[it];
    for (auto& item : externalCorrelations) {
      correlationHandler(
          item.first,
          item.second,
          static_cast<RocLogger::CorrelationDomain>(it));
    }
    std::lock_guard<std::mutex> lock(d->externalCorrelationsMutex_);
    externalCorrelations.clear();
  }

  // All Runtime API Calls
  for (auto& item : d->rows_) {
    bool filtered = false;
    if (item->type != ROCTRACER_ACTIVITY_ASYNC &&
        !isLogged(ActivityType::CUDA_RUNTIME)) {
      filtered = true;
    } else {
      switch (reinterpret_cast<rocprofAsyncRow*>(item)->domain) {
        case ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY:
          if (!isLogged(ActivityType::GPU_MEMCPY))
            filtered = true;
          break;
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        default:
          if (!isLogged(ActivityType::CONCURRENT_KERNEL))
            filtered = true;
          break;
      }
    }
    if (!filtered) {
      // Convert the begin and end timestamps from monotonic clock to system
      // clock.
      item->begin = item->begin + toffset_;
      item->end = item->end + toffset_;
      handler(item);
      ++count;
    }
  }
  return count;
}

void RocprofActivityApi::clearActivities() {
  d->clearLogs();
}

void RocprofActivityApi::enableActivities(
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

void RocprofActivityApi::disableActivities(
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
