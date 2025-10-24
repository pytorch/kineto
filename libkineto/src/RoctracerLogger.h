/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <roctracer_roctx.h>

#include "RocLogger.h"

// Local copy of hip op types.  These are public (and stable) in later rocm
// releases
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

typedef enum { HIP_OP_BARRIER_KIND_UNKNOWN_ = 0 } hip_op_barrier_kind_t_;
// end hip op defines

namespace onnxruntime {
namespace profiling {
class RocmProfiler;
}
} // namespace onnxruntime

namespace libkineto {
class RoctracerActivityApi;
}

class RoctracerApiIdList : public ApiIdList {
 public:
  uint32_t mapName(const std::string& apiName) override;
};

class RoctracerLogger {
 public:
  RoctracerLogger();
  RoctracerLogger(const RoctracerLogger&) = delete;
  RoctracerLogger& operator=(const RoctracerLogger&) = delete;

  virtual ~RoctracerLogger();

  static RoctracerLogger& singleton();

  static void pushCorrelationID(uint64_t id, RocLogger::CorrelationDomain type);
  static void popCorrelationID(RocLogger::CorrelationDomain type);

  void startLogging();
  void stopLogging();
  void clearLogs();
  void setMaxEvents(uint32_t maxBufferSize);

 private:
  bool registered_{false};
  void endTracing();

  roctracer_pool_t* hccPool_{NULL};
  static void insert_row_to_buffer(rocprofBase* row);
  static void api_callback(
      uint32_t domain,
      uint32_t cid,
      const void* callback_data,
      void* arg);
  static void activity_callback(const char* begin, const char* end, void* arg);

  RoctracerApiIdList loggedIds_;

  // Api callback data
  uint32_t maxBufferSize_{5000000}; // 5M GPU runtime/kernel events.
  std::vector<rocprofBase*> rows_;
  std::mutex rowsMutex_;

  // This vector collects pairs of correlationId and their respective
  // externalCorrelationId for each CorrelationDomain. This will be used
  // to populate the Correlation maps during post processing.
  std::vector<std::pair<uint64_t, uint64_t>>
      externalCorrelations_[RocLogger::CorrelationDomain::size];
  std::mutex externalCorrelationsMutex_;

  bool externalCorrelationEnabled_{true};
  bool logging_{false};

  friend class onnxruntime::profiling::RocmProfiler;
  friend class libkineto::RoctracerActivityApi;
};
