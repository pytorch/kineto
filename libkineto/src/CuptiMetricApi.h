/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

#include <map>
#include <vector>

#include "SampleListener.h"

namespace KINETO_NAMESPACE {

// C++ interface to CUPTI Metrics C API.
// Virtual methods are here mainly to allow easier testing.
class CuptiMetricApi {
 public:
  explicit CuptiMetricApi(CUdevice device) : device_(device) {}
  virtual ~CuptiMetricApi() {}

  virtual CUpti_MetricID idFromName(const std::string& name);
  virtual std::map<CUpti_EventID, std::string> events(CUpti_MetricID metric_id);

  virtual CUpti_MetricValueKind valueKind(CUpti_MetricID metric);
  virtual CUpti_MetricEvaluationMode evaluationMode(CUpti_MetricID metric);

  virtual SampleValue calculate(
      CUpti_MetricID metric,
      CUpti_MetricValueKind kind,
      std::vector<CUpti_EventID>& events,
      std::vector<int64_t>& values,
      int64_t duration);

 private:
  CUdevice device_;
};

} // namespace KINETO_NAMESPACE
