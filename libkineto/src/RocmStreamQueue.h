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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Demangle.h"
#include "RocLogger.h"

namespace KINETO_NAMESPACE {
namespace detail {

inline uint64_t streamIdFromHipStream(hipStream_t stream) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
}

inline uint64_t runtimeKernelStreamId(const rocprofBase* item) {
  if (item->type == ROCTRACER_ACTIVITY_KERNEL) {
    return streamIdFromHipStream(reinterpret_cast<const rocprofKernelRow*>(item)->stream);
  }
  return 0;
}

inline uint64_t runtimeCopyStreamId(const rocprofBase* item) {
  if (item->type == ROCTRACER_ACTIVITY_COPY) {
    return streamIdFromHipStream(reinterpret_cast<const rocprofCopyRow*>(item)->stream);
  }
  return 0;
}

inline uint64_t runtimeStreamId(const rocprofBase* item) {
  if (const uint64_t streamId = runtimeKernelStreamId(item)) {
    return streamId;
  }
  return runtimeCopyStreamId(item);
}

inline std::string canonicalKernelName(const std::string& name) {
  constexpr char kRocmCloneSuffix[] = " [clone .kd]";
  constexpr size_t kRocmCloneSuffixLength = sizeof(kRocmCloneSuffix) - 1;
  if (name.size() >= kRocmCloneSuffixLength &&
      name.compare(name.size() - kRocmCloneSuffixLength, kRocmCloneSuffixLength, kRocmCloneSuffix) == 0) {
    return name.substr(0, name.size() - kRocmCloneSuffixLength);
  }
  return name;
}

struct StreamQueueMaps {
  std::unordered_map<uint64_t, uint64_t> runtimeCopyStreamByCorrelation;
  std::unordered_map<uint64_t, uint64_t> runtimeKernelStreamByCorrelation;
  std::unordered_map<uint64_t, uint64_t> runtimeKernelStreamByThread;
  std::unordered_map<std::string, uint64_t> runtimeKernelStreamByName;
  std::unordered_map<uint64_t, uint64_t> asyncQueueByRuntimeStream;
  std::unordered_set<uint64_t> duplicatedAsyncKernelCorrelations;
  // Prefer preserving stream 0 over guessing when one HIP stream maps to
  // multiple ROCm async queues in the same trace.
  std::unordered_set<uint64_t> ambiguousRuntimeStreams;
  std::unordered_set<uint64_t> ambiguousRuntimeKernelThreads;
  std::unordered_set<std::string> ambiguousRuntimeKernelNames;
};

template <class Key, class Value>
inline void rememberUniqueMapping(std::unordered_map<Key, Value>& mapping,
                                  std::unordered_set<Key>& ambiguousKeys,
                                  const Key& key,
                                  Value value) {
  if (ambiguousKeys.count(key) > 0) {
    return;
  }
  const auto [it, inserted] = mapping.emplace(key, value);
  if (!inserted && it->second != value) {
    mapping.erase(it);
    ambiguousKeys.insert(key);
  }
}

inline void rememberDuplicateCorrelation(std::unordered_set<uint64_t>& seen,
                                         std::unordered_set<uint64_t>& duplicated,
                                         uint64_t correlationId) {
  if (!seen.insert(correlationId).second) {
    duplicated.insert(correlationId);
  }
}

inline std::string runtimeKernelName(const rocprofKernelRow& kernel) {
  if (!kernel.kernelName.empty()) {
    return kernel.kernelName;
  }
  if (kernel.functionAddr != nullptr) {
    return demangle(hipKernelNameRefByPtr(kernel.functionAddr, kernel.stream));
  }
  if (kernel.function != nullptr) {
    return demangle(hipKernelNameRef(kernel.function));
  }
  return "";
}

template <class IsAsyncKernel>
inline StreamQueueMaps buildStreamQueueMaps(const std::vector<rocprofBase*>& rows,
                                            bool includeCopies,
                                            bool includeKernels,
                                            IsAsyncKernel isAsyncKernel) {
  StreamQueueMaps maps;
  std::unordered_map<uint64_t, uint64_t> asyncQueueByCorrelation;
  std::unordered_set<uint64_t> ambiguousCorrelations;
  std::unordered_set<uint64_t> asyncKernelCorrelations;
  std::unordered_set<uint64_t> namedAsyncKernelCorrelations;

  for (const auto* item : rows) {
    if (includeCopies) {
      if (const uint64_t streamId = runtimeCopyStreamId(item)) {
        maps.runtimeCopyStreamByCorrelation[item->id] = streamId;
      }
    }
    if (includeKernels) {
      if (const uint64_t streamId = runtimeKernelStreamId(item)) {
        maps.runtimeKernelStreamByCorrelation[item->id] = streamId;
        const auto* kernel = reinterpret_cast<const rocprofKernelRow*>(item);
        if (kernel->tid != 0) {
          rememberUniqueMapping(maps.runtimeKernelStreamByThread,
                                maps.ambiguousRuntimeKernelThreads,
                                static_cast<uint64_t>(kernel->tid),
                                streamId);
        }
      }
    }

    if (item->type == ROCTRACER_ACTIVITY_ASYNC) {
      const auto* async = reinterpret_cast<const rocprofAsyncRow*>(item);
      if (includeCopies && async->queue != 0) {
        rememberUniqueMapping(asyncQueueByCorrelation, ambiguousCorrelations, async->id, async->queue);
      }
      if (includeKernels && isAsyncKernel(*async)) {
        rememberDuplicateCorrelation(asyncKernelCorrelations, maps.duplicatedAsyncKernelCorrelations, async->id);
        if (!async->kernelName.empty()) {
          namedAsyncKernelCorrelations.insert(async->id);
        }
      }
    }
  }

  const bool buildCopyQueueMap = includeCopies && !asyncQueueByCorrelation.empty();
  bool buildKernelNameMap = false;
  if (includeKernels) {
    for (const uint64_t correlation : maps.duplicatedAsyncKernelCorrelations) {
      if (namedAsyncKernelCorrelations.count(correlation) > 0) {
        buildKernelNameMap = true;
        break;
      }
    }
  }
  if (!buildCopyQueueMap && !buildKernelNameMap) {
    return maps;
  }

  for (const auto* item : rows) {
    if (buildCopyQueueMap) {
      const uint64_t streamId = runtimeStreamId(item);
      if (streamId != 0 && ambiguousCorrelations.count(item->id) == 0) {
        const auto queue = asyncQueueByCorrelation.find(item->id);
        if (queue != asyncQueueByCorrelation.end()) {
          rememberUniqueMapping(maps.asyncQueueByRuntimeStream, maps.ambiguousRuntimeStreams, streamId, queue->second);
        }
      }
    }
    if (buildKernelNameMap && item->type == ROCTRACER_ACTIVITY_KERNEL) {
      const auto* kernel = reinterpret_cast<const rocprofKernelRow*>(item);
      if (const uint64_t streamId = streamIdFromHipStream(kernel->stream)) {
        const auto kernelName = canonicalKernelName(runtimeKernelName(*kernel));
        if (!kernelName.empty()) {
          rememberUniqueMapping(maps.runtimeKernelStreamByName, maps.ambiguousRuntimeKernelNames, kernelName, streamId);
        }
      }
    }
  }

  return maps;
}

template <class IsAsyncCopy>
void backfillAsyncCopyStreams(std::vector<rocprofBase*>& rows, const StreamQueueMaps& maps, IsAsyncCopy isAsyncCopy) {
  if (maps.runtimeCopyStreamByCorrelation.empty() || maps.asyncQueueByRuntimeStream.empty()) {
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
    const auto stream = maps.runtimeCopyStreamByCorrelation.find(async->id);
    if (stream == maps.runtimeCopyStreamByCorrelation.end()) {
      continue;
    }
    const auto queue = maps.asyncQueueByRuntimeStream.find(stream->second);
    if (queue != maps.asyncQueueByRuntimeStream.end()) {
      async->queue = queue->second;
    }
  }
}

template <class IsAsyncKernel>
void backfillAsyncKernelStreams(std::vector<rocprofBase*>& rows,
                                const StreamQueueMaps& maps,
                                IsAsyncKernel isAsyncKernel) {
  if (maps.runtimeKernelStreamByCorrelation.empty() && maps.runtimeKernelStreamByThread.empty() &&
      maps.runtimeKernelStreamByName.empty()) {
    return;
  }
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (!isAsyncKernel(*async)) {
      continue;
    }
    const bool duplicatedCorrelation = maps.duplicatedAsyncKernelCorrelations.count(async->id) > 0;
    if (duplicatedCorrelation && !async->kernelName.empty()) {
      const auto kernelName = canonicalKernelName(async->kernelName);
      const auto stream = maps.runtimeKernelStreamByName.find(kernelName);
      if (stream != maps.runtimeKernelStreamByName.end()) {
        async->stream = stream->second;
        continue;
      }
    }
    if (async->tid != 0) {
      const auto stream = maps.runtimeKernelStreamByThread.find(async->tid);
      if (stream != maps.runtimeKernelStreamByThread.end()) {
        async->stream = stream->second;
        continue;
      }
    }
    if (duplicatedCorrelation) {
      continue;
    }
    if (async->stream != 0) {
      continue;
    }
    const auto stream = maps.runtimeKernelStreamByCorrelation.find(async->id);
    if (stream != maps.runtimeKernelStreamByCorrelation.end()) {
      async->stream = stream->second;
    }
  }
}

template <class IsAsyncCopy, class IsAsyncKernel>
void backfillAsyncStreams(std::vector<rocprofBase*>& rows,
                          bool includeCopies,
                          bool includeKernels,
                          IsAsyncCopy isAsyncCopy,
                          IsAsyncKernel isAsyncKernel) {
  if (!includeCopies && !includeKernels) {
    return;
  }
  const auto maps = buildStreamQueueMaps(rows, includeCopies, includeKernels, isAsyncKernel);
  if (includeCopies) {
    backfillAsyncCopyStreams(rows, maps, isAsyncCopy);
  }
  if (includeKernels) {
    backfillAsyncKernelStreams(rows, maps, isAsyncKernel);
  }
}

} // namespace detail
} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
