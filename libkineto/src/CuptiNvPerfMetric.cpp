/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime_api.h>
// On CUDA 13.x the host-side NVPW_MetricsContext_* family was removed from
// nvperf_host.h. We rewrote the helpers below to use NVPW_MetricsEvaluator_*
// (still present in 13.x). The gate now stays active for any CUDA >= 10.01.
#if defined(USE_CUPTI_RANGE_PROFILER) && defined(CUDART_VERSION) && \
    CUDART_VERSION > 10000
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#endif // cuda version > 10.00

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiNvPerfMetric.h"
#include "Logger.h"
#include "ScopeExit.h"

namespace KINETO_NAMESPACE {

// Add a namespace to isolate these utility functions that are only
// going to be used by the CuptiRangeProfiler. These included calls
// to NVIDIA PerfWorks APIs.
namespace nvperf {

// Implementation modeled on NVIDIA CUPTI samples at
//   /usr/local/cuda-13.2/extras/CUPTI/samples/extensions/src/profilerhost_util/{Metric,Eval}.cpp
// which uses the new NVPW_MetricsEvaluator_* family to drive the *old*
// cuptiProfiler* session API. This is the supported way to keep the legacy
// Range Profiler integration working on CUDA 13.x without porting the
// session lifecycle to cuptiRangeProfiler*.

#if defined(USE_CUPTI_RANGE_PROFILER) && defined(CUDART_VERSION) && \
    CUDART_VERSION > 10000

// Helper: expand user-supplied metric names (e.g. "smsp__warps_launched.avg")
// into the underlying raw NVPW counter dependencies and pack them into the
// NVPA_RawMetricRequest list that RawMetricsConfig / CounterDataBuilder
// consume. Uses NVPW_MetricsEvaluator_* (replacement for the gone
// NVPW_MetricsContext_*).
bool getRawMetricRequests(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<std::string>& rawMetricsDeps,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    const uint8_t* counterAvailabilityImage) {
  const bool isolated = true;
  const bool keepInstances = true;

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratchSizeParams{};
  scratchSizeParams.structSize =
      NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE;
  scratchSizeParams.pChipName = chipName.c_str();
  scratchSizeParams.pCounterAvailabilityImage = counterAvailabilityImage;
  if (!NVPW_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
          &scratchSizeParams))) {
    return false;
  }

  std::vector<uint8_t> scratchBuffer(scratchSizeParams.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params evalInitParams{};
  evalInitParams.structSize =
      NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE;
  evalInitParams.scratchBufferSize = scratchBuffer.size();
  evalInitParams.pScratchBuffer = scratchBuffer.data();
  evalInitParams.pChipName = chipName.c_str();
  evalInitParams.pCounterAvailabilityImage = counterAvailabilityImage;
  if (!NVPW_CALL(NVPW_CUDA_MetricsEvaluator_Initialize(&evalInitParams))) {
    return false;
  }
  NVPW_MetricsEvaluator* metricsEvaluator = evalInitParams.pMetricsEvaluator;

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams{};
  metricEvaluatorDestroyParams.structSize =
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricsEvaluator;
  SCOPE_EXIT([&]() {
    NVPW_MetricsEvaluator_Destroy(
        (NVPW_MetricsEvaluator_Destroy_Params*)&metricEvaluatorDestroyParams);
  });

  // For each user metric: convert name to eval request, then get raw deps.
  for (const auto& metricName : metricNames) {
    NVPW_MetricEvalRequest metricEvalRequest{};
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertParams{};
    convertParams.structSize =
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE;
    convertParams.pMetricsEvaluator = metricsEvaluator;
    convertParams.pMetricName = metricName.c_str();
    convertParams.pMetricEvalRequest = &metricEvalRequest;
    convertParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    if (!NVPW_CALL(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
            &convertParams))) {
      LOG(WARNING) << "Failed to convert metric name to eval request: "
                   << metricName;
      return false;
    }

    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params depsParams{};
    depsParams.structSize =
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE;
    depsParams.pMetricsEvaluator = metricsEvaluator;
    depsParams.pMetricEvalRequests = &metricEvalRequest;
    depsParams.numMetricEvalRequests = 1;
    depsParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    depsParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
    if (!NVPW_CALL(NVPW_MetricsEvaluator_GetMetricRawDependencies(&depsParams))) {
      return false;
    }

    std::vector<const char*> rawDeps(depsParams.numRawDependencies);
    depsParams.ppRawDependencies = rawDeps.data();
    if (!NVPW_CALL(NVPW_MetricsEvaluator_GetMetricRawDependencies(&depsParams))) {
      return false;
    }

    for (size_t i = 0; i < rawDeps.size(); ++i) {
      rawMetricsDeps.emplace_back(rawDeps[i]);
    }
  }

  for (const auto& rawMetricName : rawMetricsDeps) {
    NVPA_RawMetricRequest metricRequest{};
    metricRequest.structSize = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
    metricRequest.pMetricName = rawMetricName.c_str();
    metricRequest.isolated = isolated;
    metricRequest.keepInstances = keepInstances;
    rawMetricRequests.push_back(metricRequest);
    VLOG(1) << "Adding raw metric struct  : raw metric = " << rawMetricName
            << " isolated = " << isolated << " keepinst = " << keepInstances;
  }

  if (rawMetricRequests.empty()) {
    LOG(WARNING) << "CUPTI Profiler was unable to configure any metrics";
    return false;
  }
  return true;
}

bool getProfilerConfigImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* counterAvailabilityImage) {
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> rawMetricDeps;
  if (!getRawMetricRequests(
          chipName, metricNames, rawMetricDeps, rawMetricRequests,
          counterAvailabilityImage)) {
    return false;
  }

  NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams{};
  rawMetricsConfigCreateParams.structSize =
      NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE;
  rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  rawMetricsConfigCreateParams.pChipName = chipName.c_str();
  rawMetricsConfigCreateParams.pCounterAvailabilityImage =
      counterAvailabilityImage;
  if (!NVPW_CALL(
          NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams))) {
    return false;
  }
  NVPA_RawMetricsConfig* rawMetricsConfig =
      rawMetricsConfigCreateParams.pRawMetricsConfig;

  if (counterAvailabilityImage) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams{};
    setCounterAvailabilityParams.structSize =
        NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE;
    setCounterAvailabilityParams.pRawMetricsConfig = rawMetricsConfig;
    setCounterAvailabilityParams.pCounterAvailabilityImage =
        counterAvailabilityImage;
    if (!NVPW_CALL(NVPW_RawMetricsConfig_SetCounterAvailability(
            &setCounterAvailabilityParams))) {
      return false;
    }
  }

  NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams{};
  rawMetricsConfigDestroyParams.structSize =
      NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
  rawMetricsConfigDestroyParams.pRawMetricsConfig = rawMetricsConfig;
  SCOPE_EXIT([&]() {
    NVPW_RawMetricsConfig_Destroy(
        (NVPW_RawMetricsConfig_Destroy_Params*)&rawMetricsConfigDestroyParams);
  });

  NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams{};
  beginPassGroupParams.structSize =
      NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
  beginPassGroupParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams))) {
    return false;
  }

  NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams{};
  addMetricsParams.structSize =
      NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
  addMetricsParams.pRawMetricsConfig = rawMetricsConfig;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  if (!NVPW_CALL(NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams))) {
    return false;
  }

  NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams{};
  endPassGroupParams.structSize =
      NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
  endPassGroupParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams))) {
    return false;
  }

  NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams{};
  generateConfigImageParams.structSize =
      NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE;
  generateConfigImageParams.pRawMetricsConfig = rawMetricsConfig;
  if (!NVPW_CALL(NVPW_RawMetricsConfig_GenerateConfigImage(
          &generateConfigImageParams))) {
    return false;
  }

  NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams{};
  getConfigImageParams.structSize =
      NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
  getConfigImageParams.pRawMetricsConfig = rawMetricsConfig;
  getConfigImageParams.bytesAllocated = 0;
  getConfigImageParams.pBuffer = nullptr;
  if (!NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams))) {
    return false;
  }

  configImage.resize(getConfigImageParams.bytesCopied);
  getConfigImageParams.bytesAllocated = configImage.size();
  getConfigImageParams.pBuffer = configImage.data();
  if (!NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams))) {
    return false;
  }

  return true;
}

bool getCounterDataPrefixImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataImagePrefix) {
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> rawMetricDeps;
  if (!getRawMetricRequests(
          chipName, metricNames, rawMetricDeps, rawMetricRequests,
          /*counterAvailabilityImage=*/nullptr)) {
    return false;
  }

  NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams{};
  counterDataBuilderCreateParams.structSize =
      NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE;
  counterDataBuilderCreateParams.pChipName = chipName.c_str();
  counterDataBuilderCreateParams.pCounterAvailabilityImage = nullptr;
  if (!NVPW_CALL(NVPW_CUDA_CounterDataBuilder_Create(
          &counterDataBuilderCreateParams))) {
    return false;
  }

  NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams{};
  counterDataBuilderDestroyParams.structSize =
      NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE;
  counterDataBuilderDestroyParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  SCOPE_EXIT([&]() {
    NVPW_CounterDataBuilder_Destroy(
        (NVPW_CounterDataBuilder_Destroy_Params*)&counterDataBuilderDestroyParams);
  });

  NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams{};
  addMetricsParams.structSize =
      NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE;
  addMetricsParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  if (!NVPW_CALL(NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams))) {
    return false;
  }

  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams{};
  getCounterDataPrefixParams.structSize =
      NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
  getCounterDataPrefixParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  getCounterDataPrefixParams.bytesAllocated = 0;
  getCounterDataPrefixParams.pBuffer = nullptr;
  if (!NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(
          &getCounterDataPrefixParams))) {
    return false;
  }

  counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);
  getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
  getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
  if (!NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(
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

  NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams{};
  getRangeDescParams.structSize =
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE;
  getRangeDescParams.pCounterDataImage = counterDataImage.data();
  getRangeDescParams.rangeIndex = rangeIndex;

  if (!NVPW_CALL(NVPW_Profiler_CounterData_GetRangeDescriptions(
          &getRangeDescParams))) {
    return "";
  }

  descriptionPtrs.resize(getRangeDescParams.numDescriptions);
  getRangeDescParams.ppDescriptions = descriptionPtrs.data();

  if (!NVPW_CALL(NVPW_Profiler_CounterData_GetRangeDescriptions(
          &getRangeDescParams))) {
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

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratchSizeParams{};
  scratchSizeParams.structSize =
      NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE;
  scratchSizeParams.pChipName = chipName.c_str();
  if (!NVPW_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
          &scratchSizeParams))) {
    return {};
  }

  std::vector<uint8_t> scratchBuffer(scratchSizeParams.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params evalInitParams{};
  evalInitParams.structSize =
      NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE;
  evalInitParams.scratchBufferSize = scratchBuffer.size();
  evalInitParams.pScratchBuffer = scratchBuffer.data();
  evalInitParams.pChipName = chipName.c_str();
  evalInitParams.pCounterDataImage = counterDataImage.data();
  evalInitParams.counterDataImageSize = counterDataImage.size();
  if (!NVPW_CALL(NVPW_CUDA_MetricsEvaluator_Initialize(&evalInitParams))) {
    return {};
  }
  NVPW_MetricsEvaluator* metricsEvaluator = evalInitParams.pMetricsEvaluator;

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams{};
  metricEvaluatorDestroyParams.structSize =
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricsEvaluator;
  SCOPE_EXIT([&]() {
    NVPW_MetricsEvaluator_Destroy(
        (NVPW_MetricsEvaluator_Destroy_Params*)&metricEvaluatorDestroyParams);
  });

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams{};
  getNumRangesParams.structSize = NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE;
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  if (!NVPW_CALL(NVPW_CounterData_GetNumRanges(&getNumRangesParams))) {
    return {};
  }

  std::vector<NVPW_MetricEvalRequest> metricEvalRequests(metricNames.size());
  for (size_t i = 0; i < metricNames.size(); ++i) {
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertParams{};
    convertParams.structSize =
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE;
    convertParams.pMetricsEvaluator = metricsEvaluator;
    convertParams.pMetricName = metricNames[i].c_str();
    convertParams.pMetricEvalRequest = &metricEvalRequests[i];
    convertParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    if (!NVPW_CALL(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
            &convertParams))) {
      LOG(WARNING) << "Failed to convert metric name to eval request: "
                   << metricNames[i];
      return {};
    }
  }

  CuptiProfilerResult result{};
  result.metricNames = metricNames;

  for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges;
       ++rangeIndex) {
    CuptiRangeMeasurement rangeData{};
    rangeData.rangeName = getRangeDescription(counterDataImage, rangeIndex);
    rangeData.values.resize(metricNames.size());

    NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribsParams{};
    setDeviceAttribsParams.structSize =
        NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE;
    setDeviceAttribsParams.pMetricsEvaluator = metricsEvaluator;
    setDeviceAttribsParams.pCounterDataImage = counterDataImage.data();
    setDeviceAttribsParams.counterDataImageSize = counterDataImage.size();
    if (!NVPW_CALL(NVPW_MetricsEvaluator_SetDeviceAttributes(
            &setDeviceAttribsParams))) {
      LOG(WARNING) << "Failed to set device attributes for range: "
                   << rangeData.rangeName;
      continue;
    }

    NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evalParams{};
    evalParams.structSize =
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE;
    evalParams.pMetricsEvaluator = metricsEvaluator;
    evalParams.pMetricEvalRequests = metricEvalRequests.data();
    evalParams.numMetricEvalRequests = metricEvalRequests.size();
    evalParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    evalParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
    evalParams.pCounterDataImage = counterDataImage.data();
    evalParams.counterDataImageSize = counterDataImage.size();
    evalParams.rangeIndex = rangeIndex;
    evalParams.isolated = true;
    evalParams.pMetricValues = rangeData.values.data();

    if (!NVPW_CALL(NVPW_MetricsEvaluator_EvaluateToGpuValues(&evalParams))) {
      LOG(WARNING) << "Failed to evaluate metrics for range : "
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
    [[maybe_unused]] const std::string& chipName,
    [[maybe_unused]] const std::vector<std::string>& metricNames,
    [[maybe_unused]] std::vector<uint8_t>& configImage,
    [[maybe_unused]] const uint8_t* counterAvailabilityImage) {
  return false;
}

bool getCounterDataPrefixImage(
    [[maybe_unused]] const std::string& chipName,
    [[maybe_unused]] const std::vector<std::string>& metricNames,
    [[maybe_unused]] std::vector<uint8_t>& counterDataImagePrefix) {
  return false;
}

CuptiProfilerResult evalMetricValues(
    [[maybe_unused]] const std::string& chipName,
    [[maybe_unused]] const std::vector<uint8_t>& counterDataImage,
    [[maybe_unused]] const std::vector<std::string>& metricNames,
    [[maybe_unused]] bool verbose) {
  return {};
}

#endif // cuda version > 10.00

} // namespace nvperf
} // namespace KINETO_NAMESPACE
