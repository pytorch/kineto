/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef HAS_CUPTI
#include <cuda_runtime_api.h>
#if defined(USE_CUPTI_RANGE_PROFILER) && defined(CUDART_VERSION) && CUDART_VERSION > 10000
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#endif // cuda version > 10.00 and < 11.04
#endif // HAS_CUPTI

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ScopeExit.h"
#include "CuptiNvPerfMetric.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {

// Add a namespace to isolate these utility functions that are only
// going to be used by the CuptiRangeProfiler. These included calls
// to NVIDIA PerfWorks APIs.
namespace nvperf {


// Largely based on NVIDIA sample code provided with CUDA release
//  files Metric.cpp and Eval.cpp

// -------------------------------------------------
// Metric and Counter Data Configuration
// -------------------------------------------------


// Note: Be carful before modifying the code below. There is a specific
// sequence one needs to follow to program the metrics else things may
// stop working. We tried to keep the flow consistent with the example
// code from NVIDIA. Since most of the programmability comes from
// the CUPTI profiler metric names this should be okay.

// Only supported on CUDA RT Version between 10.0 and 11.04.
// After CUDA RT 11.04, the structure has changed.
// TODO update the structure NVPA_RawMetricsConfig to support 11.04
#if defined(USE_CUPTI_RANGE_PROFILER) && defined(CUDART_VERSION) && CUDART_VERSION > 10000

bool getRawMetricRequests(
    NVPA_MetricsContext* metricsContext,
    std::vector<std::string> metricNames,
    std::vector<std::string>& rawMetricsDeps,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests) {
  bool isolated = true;
  /* Bug in collection with collection of metrics without instances, keep it
   * to true*/
  bool keepInstances = true;

  for (const auto& metricName : metricNames) {

    NVPW_MetricsContext_GetMetricProperties_Begin_Params
        getMetricPropertiesBeginParams = {
            NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    getMetricPropertiesBeginParams.pMetricsContext = metricsContext;
    getMetricPropertiesBeginParams.pMetricName = metricName.c_str();

    if (!NVPW_CALL(
        NVPW_MetricsContext_GetMetricProperties_Begin(
            &getMetricPropertiesBeginParams))) {
      return false;
    }

    for (const char** metricDepsIt =
             getMetricPropertiesBeginParams.ppRawMetricDependencies;
         *metricDepsIt;
         ++metricDepsIt) {
      rawMetricsDeps.push_back(*metricDepsIt);
    }

    NVPW_MetricsContext_GetMetricProperties_End_Params
        getMetricPropertiesEndParams = {
            NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE, nullptr, nullptr};
    getMetricPropertiesEndParams.pMetricsContext = metricsContext;

    if (!NVPW_CALL(NVPW_MetricsContext_GetMetricProperties_End(
            &getMetricPropertiesEndParams))) {
      return false;
    }
  }

  for (const auto& rawMetricName : rawMetricsDeps) {
    NVPA_RawMetricRequest metricRequest = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE, nullptr, nullptr, false, false};
    metricRequest.pMetricName = rawMetricName.c_str();
    metricRequest.isolated = isolated;
    metricRequest.keepInstances = keepInstances;
    rawMetricRequests.push_back(metricRequest);
    VLOG(1) << "Adding raw metric struct  : raw metric = " << rawMetricName
        << " isolated = " << isolated << " keepinst = " << keepInstances;
  }

  if (rawMetricRequests.size() == 0) {
    LOG(WARNING) << "CUPTI Profiler was unable to configure any metrics";
    return false;
  }
  return true;
}

// Setup CUPTI Profiler Config Image
bool getProfilerConfigImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* counterAvailabilityImage) {

  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE, nullptr, nullptr, nullptr};
  metricsContextCreateParams.pChipName = chipName.c_str();

  if (!NVPW_CALL(
        NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams))) {
    return false;
  }

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE, nullptr, nullptr};
  metricsContextDestroyParams.pMetricsContext =
      metricsContextCreateParams.pMetricsContext;

  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params*)&metricsContextDestroyParams);
  });

  // Get all raw metrics required for given metricNames list
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;

  // note: we need a variable at this functions scope to hold the string
  // pointers for underlying C char arrays.
  std::vector<std::string> rawMetricDeps;

  if (!getRawMetricRequests(
      metricsContextCreateParams.pMetricsContext,
      metricNames,
      rawMetricDeps,
      rawMetricRequests)) {
    return false;
  }

  // Starting CUDA 11.4 the metric config create call and struct has changed
#if CUDART_VERSION < 11040
   NVPA_RawMetricsConfigOptions metricsConfigOptions = {
       NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE, nullptr};
#else
   NVPW_CUDA_RawMetricsConfig_Create_Params metricsConfigOptions = {
       NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE, nullptr, NVPA_ACTIVITY_KIND_INVALID, nullptr, nullptr};
#endif // CUDART_VERSION < 11040

   metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
   metricsConfigOptions.pChipName = chipName.c_str();

   NVPA_RawMetricsConfig* rawMetricsConfig;
#if CUDART_VERSION < 11040
   if (!NVPW_CALL(
         NVPA_RawMetricsConfig_Create(
           &metricsConfigOptions, &rawMetricsConfig))) {
     return false;
   }
#else
   if (!NVPW_CALL(NVPW_CUDA_RawMetricsConfig_Create(&metricsConfigOptions))) {
     return false;
   }
  rawMetricsConfig = metricsConfigOptions.pRawMetricsConfig;
#endif // CUDART_VERSION < 11040

  // TODO check if this is required
  if (counterAvailabilityImage) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params
        setCounterAvailabilityParams = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE, nullptr, nullptr, nullptr};
    setCounterAvailabilityParams.pRawMetricsConfig = rawMetricsConfig;
    setCounterAvailabilityParams.pCounterAvailabilityImage =
        counterAvailabilityImage;
    if (!NVPW_CALL(
          NVPW_RawMetricsConfig_SetCounterAvailability(
            &setCounterAvailabilityParams))) {
      return false;
    }
  }

  NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
      NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE, nullptr, nullptr};
  rawMetricsConfigDestroyParams.pRawMetricsConfig = rawMetricsConfig;
  SCOPE_EXIT([&]() {
    NVPW_RawMetricsConfig_Destroy(
        (NVPW_RawMetricsConfig_Destroy_Params*)&rawMetricsConfigDestroyParams);
  });

  // Start a Raw Metric Pass group
  NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
      NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE, nullptr, nullptr, 0};
  beginPassGroupParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams))) {
    return false;
  }

  // Add all raw metrics
  NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
      NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE, nullptr, nullptr, nullptr, 0};
  addMetricsParams.pRawMetricsConfig = rawMetricsConfig;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams))) {
    return false;
  }

  // End pass group
  NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
      NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE, nullptr, nullptr};
  endPassGroupParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams))) {
    return false;
  }

  // Setup Config Image generation
  NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
      NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE, nullptr, nullptr, false};
  generateConfigImageParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams))) {
    return false;
  }

  // Get the Config Image size... nearly there
  NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
      NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE, nullptr, nullptr};
  getConfigImageParams.pRawMetricsConfig = rawMetricsConfig;
  getConfigImageParams.bytesAllocated = 0;
  getConfigImageParams.pBuffer = nullptr;
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams))) {
    return false;
  }

  configImage.resize(getConfigImageParams.bytesCopied);

  // Write the Config image binary
  getConfigImageParams.bytesAllocated = configImage.size();
  getConfigImageParams.pBuffer = configImage.data();
  if (!NVPW_CALL(
        NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams))) {
    return false;
  }

  return true;
}

bool getCounterDataPrefixImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataImagePrefix) {

  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE, nullptr, nullptr};
  metricsContextCreateParams.pChipName = chipName.c_str();

  if (!NVPW_CALL(
        NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams))) {
    return false;
  }

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE, nullptr, nullptr};
  metricsContextDestroyParams.pMetricsContext =
      metricsContextCreateParams.pMetricsContext;


  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params*)&metricsContextDestroyParams);
  });

  // Get all raw metrics required for given metricNames list
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;

  // note: we need a variable at this functions scope to hold the string
  // pointers for underlying C char arrays.
  std::vector<std::string> rawMetricDeps;

  if (!getRawMetricRequests(
      metricsContextCreateParams.pMetricsContext,
      metricNames,
      rawMetricDeps,
      rawMetricRequests)) {
    return false;
  }

  // Setup Counter Data builder
  NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
      NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE, nullptr, nullptr, nullptr};
  counterDataBuilderCreateParams.pChipName = chipName.c_str();
  if (!NVPW_CALL(
        NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams))) {
    return false;
  }

  NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
      NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE, nullptr, nullptr};
  counterDataBuilderDestroyParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  SCOPE_EXIT([&]() {
    NVPW_CounterDataBuilder_Destroy((
        NVPW_CounterDataBuilder_Destroy_Params*)&counterDataBuilderDestroyParams);
  });

  // Add metrics to counter data image prefix
  NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
      NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE, nullptr, nullptr, nullptr, 0};
  addMetricsParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  if (!NVPW_CALL(
        NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams))) {
    return false;
  }

  // Get image prefix size
  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params
      getCounterDataPrefixParams = {
          NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE, nullptr, nullptr, 0, nullptr, 0};
  getCounterDataPrefixParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  getCounterDataPrefixParams.bytesAllocated = 0;
  getCounterDataPrefixParams.pBuffer = nullptr;
  if (!NVPW_CALL(
        NVPW_CounterDataBuilder_GetCounterDataPrefix(
          &getCounterDataPrefixParams))) {
    return false;
  }

  counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

  // Now write counter data image prefix
  getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
  getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
  if (!NVPW_CALL(
        NVPW_CounterDataBuilder_GetCounterDataPrefix(
          &getCounterDataPrefixParams))) {
    return false;
  }

  return true;
}

// -------------------------------------------------
// Metric and Counter Evaluation Utilities
// -------------------------------------------------

std::string getRangeDescription(
    const std::vector<uint8_t>& counterDataImage,
    int rangeIndex) {
  std::vector<const char*> descriptionPtrs;

  NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = {
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE, nullptr};
  getRangeDescParams.pCounterDataImage = counterDataImage.data();
  getRangeDescParams.rangeIndex = rangeIndex;

  if (!NVPW_CALL(
      NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams))) {
    return "";
  }

  descriptionPtrs.resize(getRangeDescParams.numDescriptions);
  getRangeDescParams.ppDescriptions = descriptionPtrs.data();

  if (!NVPW_CALL(
      NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams))) {
    return "";
  }

  std::string rangeName;

  for (size_t i = 0; i < getRangeDescParams.numDescriptions; i++) {
    if (i > 0) {
      rangeName.append("/");
    }
    rangeName.append(descriptionPtrs[i]);
  }
  return rangeName;
}

CuptiProfilerResult evalMetricValues(
    const std::string& chipName,
    const std::vector<uint8_t>& counterDataImage,
    const std::vector<std::string>& metricNames,
    bool verbose) {

  if (!counterDataImage.size()) {
    LOG(ERROR) << "Counter Data Image is empty!";
    return {};
  }

  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE, nullptr};
  metricsContextCreateParams.pChipName = chipName.c_str();
  if (!NVPW_CALL(
        NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams))) {
    return {};
  }

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE, nullptr};
  metricsContextDestroyParams.pMetricsContext =
      metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params*)&metricsContextDestroyParams);
  });

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE, nullptr};
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  if (!NVPW_CALL(
      NVPW_CounterData_GetNumRanges(&getNumRangesParams))) {
    return {};
  }

  // TBD in the future support special chars in metric name
  // for now these are default
  const bool isolated = true;

  // API takes a 2D array of chars
  std::vector<const char*> metricNamePtrs;

  for (const auto& metric : metricNames) {
    metricNamePtrs.push_back(metric.c_str());
  }

  CuptiProfilerResult result{
    .metricNames = metricNames};

  for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges;
       ++rangeIndex) {

    CuptiRangeMeasurement rangeData {
    .rangeName = getRangeDescription(counterDataImage, rangeIndex)};
    rangeData.values.resize(metricNames.size());

    // First set Counter data image with current range
    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE, nullptr};

    setCounterDataParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;
    setCounterDataParams.pCounterDataImage = counterDataImage.data();
    setCounterDataParams.isolated = isolated;
    setCounterDataParams.rangeIndex = rangeIndex;

    NVPW_CALL(NVPW_MetricsContext_SetCounterData(&setCounterDataParams));


    // Now we can evaluate GPU metrics
    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
        NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE, nullptr};
    evalToGpuParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;
    evalToGpuParams.numMetrics = metricNamePtrs.size();
    evalToGpuParams.ppMetricNames = metricNamePtrs.data();
    evalToGpuParams.pMetricValues = rangeData.values.data();

    if (!NVPW_CALL(NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams))) {
      LOG(WARNING) << "Failed to evaluate metris for range : "
                   << rangeData.rangeName;
      continue;
    }

    if (verbose) {
      for (size_t i = 0; i < metricNames.size(); i++) {
        LOG(INFO) << "rangeName: " << rangeData.rangeName
                  << "\tmetricName: " << metricNames[i]
                  << "\tgpuValue: " << rangeData.values[i];
      }
    }

    result.rangeVals.emplace_back(std::move(rangeData));
  }

  return result;
}

#else

bool getProfilerConfigImage(
    const std::string& /*chipName*/,
    const std::vector<std::string>& /*metricNames*/,
    std::vector<uint8_t>& /*configImage*/,
    const uint8_t* /*counterAvailabilityImage*/) {
  return false;
}

bool getCounterDataPrefixImage(
    const std::string& /*chipName*/,
    const std::vector<std::string>& /*metricNames*/,
    std::vector<uint8_t>& /*counterDataImagePrefix*/) {
  return false;
}

CuptiProfilerResult evalMetricValues(
    const std::string& /*chipName*/,
    const std::vector<uint8_t>& /*counterDataImage*/,
    const std::vector<std::string>& /*metricNames*/,
    bool /*verbose*/) {
  return {};
}

#endif // cuda version > 10.00 and < 11.04

} // namespace nvperf
} // namespace KINETO_NAMESPACE
