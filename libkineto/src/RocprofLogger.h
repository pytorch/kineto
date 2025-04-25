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

#include <rocprofiler-sdk/registration.h>

#include "RocLogger.h"

class RocprofLogger {
 public:
  RocprofLogger();
  RocprofLogger(const RocprofLogger&) = delete;
  RocprofLogger& operator=(const RocprofLogger&) = delete;

  virtual ~RocprofLogger();

  static RocprofLogger& singleton();

  static void pushCorrelationID(uint64_t id, RocLogger::CorrelationDomain type);
  static void popCorrelationID(RocLogger::CorrelationDomain type);

  void startLogging();
  void stopLogging();
  void clearLogs();

  static int toolInit(
      rocprofiler_client_finalize_t finalize_func,
      void* tool_data);
  static void toolFinialize(void* tool_data);

  static std::string opString(
      rocprofiler_callback_tracing_kind_t kind,
      rocprofiler_tracing_operation_t op);

 private:
  bool registered_{false};
  void endTracing();

  static void insert_row_to_buffer(rocprofBase* row);

  //
  static void api_callback(
      rocprofiler_callback_tracing_record_t record,
      rocprofiler_user_data_t* user_data,
      void* callback_data);
  static void code_object_callback(
      rocprofiler_callback_tracing_record_t record,
      rocprofiler_user_data_t* user_data,
      void* callback_data);

  // Api callback data
  uint32_t maxBufferSize_{1000000}; // 1M GPU runtime/kernel events.
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

  friend class libkineto::RocprofActivityApi;
};
