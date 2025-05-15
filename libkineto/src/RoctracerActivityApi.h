/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#ifdef HAS_ROCTRACER

#include <atomic>
#include <functional>
#include <set>

#include <roctracer.h>
#include "RoctracerLogger.h"

#include "ActivityType.h"
#include "GenericTraceActivity.h"

class RoctracerLogger;

namespace KINETO_NAMESPACE {

using namespace libkineto;

class RoctracerActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  RoctracerActivityApi();
  RoctracerActivityApi(const RoctracerActivityApi&) = delete;
  RoctracerActivityApi& operator=(const RoctracerActivityApi&) = delete;

  virtual ~RoctracerActivityApi();

  static RoctracerActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(const std::set<ActivityType>& selected_activities);
  void disableActivities(const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void teardownContext() {}
  void setMaxEvents(uint32_t maxEvents);

  virtual int processActivities(
      std::function<void(const roctracerBase*)> handler,
      std::function<
          void(uint64_t, uint64_t, RoctracerLogger::CorrelationDomain)>
          correlationHandler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};

 private:
  bool registered_{false};

  // Enabled Activity Filters
  uint32_t activityMask_{0};
  uint32_t activityMaskSnapshot_{0};
  bool isLogged(libkineto::ActivityType atype) const;

  RoctracerLogger* d;
};

} // namespace KINETO_NAMESPACE
#endif
