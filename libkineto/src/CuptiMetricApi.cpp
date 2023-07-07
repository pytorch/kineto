/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiMetricApi.h"

#include <chrono>

#include "Logger.h"
#include "cupti_call.h"

using std::vector;

namespace KINETO_NAMESPACE {

CUpti_MetricID CuptiMetricApi::idFromName(const std::string& name) {
  CUpti_MetricID metric_id{~0u};
  CUptiResult res =
      CUPTI_CALL(cuptiMetricGetIdFromName(device_, name.c_str(), &metric_id));
  if (res == CUPTI_ERROR_INVALID_METRIC_NAME) {
    LOG(WARNING) << "Invalid metric name: " << name;
  }
  return metric_id;
}

// Return a map of event IDs and names for a given metric id.
// Note that many events don't have a name. In that case the name will
// be set to the empty string.
std::map<CUpti_EventID, std::string> CuptiMetricApi::events(
    CUpti_MetricID metric_id) {
  uint32_t num_events = 0;
  CUPTI_CALL(cuptiMetricGetNumEvents(metric_id, &num_events));
  vector<CUpti_EventID> ids(num_events);
  size_t array_size = num_events * sizeof(CUpti_EventID);
  CUPTI_CALL(cuptiMetricEnumEvents(metric_id, &array_size, ids.data()));
  std::map<CUpti_EventID, std::string> res;
  for (CUpti_EventID id : ids) {
    // Attempt to lookup name from CUPTI
    constexpr size_t kMaxEventNameLength = 64;
    char cupti_name[kMaxEventNameLength];
    size_t size = kMaxEventNameLength;
    CUPTI_CALL(
        cuptiEventGetAttribute(id, CUPTI_EVENT_ATTR_NAME, &size, cupti_name));
    cupti_name[kMaxEventNameLength - 1] = 0;

    // CUPTI "helpfully" returns "event_name" when the event is unnamed.
    if (size > 0 && strcmp(cupti_name, "event_name") != 0) {
      res.emplace(id, cupti_name);
    } else {
      res.emplace(id, "");
    }
  }
  return res;
}

CUpti_MetricValueKind CuptiMetricApi::valueKind(CUpti_MetricID metric) {
  CUpti_MetricValueKind res{CUPTI_METRIC_VALUE_KIND_FORCE_INT};
  size_t value_kind_size = sizeof(res);
  CUPTI_CALL(cuptiMetricGetAttribute(
      metric, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_size, &res));
  return res;
}

CUpti_MetricEvaluationMode CuptiMetricApi::evaluationMode(
    CUpti_MetricID metric) {
  CUpti_MetricEvaluationMode eval_mode{
      CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE};
  size_t eval_mode_size = sizeof(eval_mode);
  CUPTI_CALL(cuptiMetricGetAttribute(
      metric, CUPTI_METRIC_ATTR_EVALUATION_MODE, &eval_mode_size, &eval_mode));
  return eval_mode;
}

// FIXME: Consider caching value kind here
SampleValue CuptiMetricApi::calculate(
    CUpti_MetricID metric,
    CUpti_MetricValueKind kind,
    vector<CUpti_EventID>& events,
    vector<int64_t>& values,
    int64_t duration) {
  CUpti_MetricValue metric_value;
  CUPTI_CALL(cuptiMetricGetValue(
      device_,
      metric,
      events.size() * sizeof(CUpti_EventID),
      events.data(),
      values.size() * sizeof(int64_t),
      reinterpret_cast<uint64_t*>(values.data()),
      duration,
      &metric_value));

  switch (kind) {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
    case CUPTI_METRIC_VALUE_KIND_PERCENT:
      return SampleValue(metric_value.metricValueDouble);
    case CUPTI_METRIC_VALUE_KIND_UINT64:
    case CUPTI_METRIC_VALUE_KIND_INT64:
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
      return SampleValue(metric_value.metricValueUint64);
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
      return SampleValue((int)metric_value.metricValueUtilizationLevel);
    default:
      assert(false);
  }
  return SampleValue(-1);
}

} // namespace KINETO_NAMESPACE
