/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <functional>

#ifdef HAS_ROCTRACER
#include <roctracer.h>
#include "RoctracerLogger.h"
#endif

#include "ActivityType.h"
#include "GenericTraceActivity.h"

class RoctracerLogger;

namespace KINETO_NAMESPACE {

using namespace libkineto;

class RoctracerActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  RoctracerActivityApi();
  RoctracerActivityApi(const RoctracerActivityApi&) = delete;
  RoctracerActivityApi& operator=(const RoctracerActivityApi&) = delete;

  virtual ~RoctracerActivityApi();

  static RoctracerActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(
    const std::set<ActivityType>& selected_activities);
  void disableActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void teardownContext() {}

  int processActivities(std::function<void(const roctracerBase*)> handler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};

 private:
  bool registered_{false};

  // Enabled Activity Filters
  uint32_t activityMask_{0};
  uint32_t activityMaskSnapshot_{0};
  bool isLogged(libkineto::ActivityType atype);

  RoctracerLogger *d;
};

} // namespace KINETO_NAMESPACE
