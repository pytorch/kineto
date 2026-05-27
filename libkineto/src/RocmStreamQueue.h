/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef HAS_ROCTRACER

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "RocLogger.h"

namespace KINETO_NAMESPACE {
namespace detail {

inline uint64_t streamIdFromHipStream(hipStream_t stream) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
}

inline uint64_t runtimeStreamId(const rocprofBase* item) {
  if (item->type == ROCTRACER_ACTIVITY_KERNEL) {
    return streamIdFromHipStream(reinterpret_cast<const rocprofKernelRow*>(item)->stream);
  }
  if (item->type == ROCTRACER_ACTIVITY_COPY) {
    return streamIdFromHipStream(reinterpret_cast<const rocprofCopyRow*>(item)->stream);
  }
  return 0;
}

struct StreamQueueMaps {
  std::unordered_map<uint64_t, uint64_t> runtimeStreamByCorrelation;
  std::unordered_map<uint64_t, uint64_t> asyncQueueByRuntimeStream;
  // Prefer preserving stream 0 over guessing when one HIP stream maps to
  // multiple ROCm async queues in the same trace.
  std::unordered_set<uint64_t> ambiguousRuntimeStreams;
};

inline void rememberCorrelationQueue(std::unordered_map<uint64_t, uint64_t>& asyncQueueByCorrelation,
                                     std::unordered_set<uint64_t>& ambiguousCorrelations,
                                     uint64_t correlationId,
                                     uint64_t queue) {
  if (ambiguousCorrelations.count(correlationId) > 0) {
    return;
  }
  const auto [queueIt, inserted] = asyncQueueByCorrelation.emplace(correlationId, queue);
  if (!inserted && queueIt->second != queue) {
    asyncQueueByCorrelation.erase(queueIt);
    ambiguousCorrelations.insert(correlationId);
  }
}

inline void rememberQueueForRuntimeStream(StreamQueueMaps& maps, uint64_t stream, uint64_t queue) {
  if (maps.ambiguousRuntimeStreams.count(stream) > 0) {
    return;
  }
  const auto [queueIt, inserted] = maps.asyncQueueByRuntimeStream.emplace(stream, queue);
  if (!inserted && queueIt->second != queue) {
    maps.asyncQueueByRuntimeStream.erase(queueIt);
    maps.ambiguousRuntimeStreams.insert(stream);
  }
}

inline StreamQueueMaps buildStreamQueueMaps(const std::vector<rocprofBase*>& rows) {
  StreamQueueMaps maps;
  std::unordered_map<uint64_t, uint64_t> asyncQueueByCorrelation;
  std::unordered_set<uint64_t> ambiguousCorrelations;

  for (const auto* item : rows) {
    const uint64_t streamId = runtimeStreamId(item);
    if (streamId != 0) {
      maps.runtimeStreamByCorrelation[item->id] = streamId;
    }

    if (item->type == ROCTRACER_ACTIVITY_ASYNC) {
      const auto* async = reinterpret_cast<const rocprofAsyncRow*>(item);
      if (async->queue != 0) {
        rememberCorrelationQueue(asyncQueueByCorrelation, ambiguousCorrelations, async->id, async->queue);
      }
    }
  }

  for (const auto* item : rows) {
    const uint64_t streamId = runtimeStreamId(item);
    if (streamId == 0 || ambiguousCorrelations.count(item->id) > 0) {
      continue;
    }
    const auto queue = asyncQueueByCorrelation.find(item->id);
    if (queue != asyncQueueByCorrelation.end()) {
      rememberQueueForRuntimeStream(maps, streamId, queue->second);
    }
  }

  return maps;
}

template <class IsAsyncCopy>
void backfillAsyncCopyStreams(std::vector<rocprofBase*>& rows, IsAsyncCopy isAsyncCopy) {
  const auto maps = buildStreamQueueMaps(rows);
  if (maps.runtimeStreamByCorrelation.empty() || maps.asyncQueueByRuntimeStream.empty()) {
    return;
  }
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (!isAsyncCopy(*async) || async->queue != 0) {
      continue;
    }
    const auto stream = maps.runtimeStreamByCorrelation.find(async->id);
    if (stream == maps.runtimeStreamByCorrelation.end() || maps.ambiguousRuntimeStreams.count(stream->second) > 0) {
      continue;
    }
    const auto queue = maps.asyncQueueByRuntimeStream.find(stream->second);
    if (queue != maps.asyncQueueByRuntimeStream.end()) {
      async->queue = queue->second;
    }
  }
}

} // namespace detail
} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
