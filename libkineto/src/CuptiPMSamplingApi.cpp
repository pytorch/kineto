/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiPMSamplingApi.h"

#include <stdexcept>

#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_target.h>

#include "DeviceUtil.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {
namespace {

constexpr size_t kHardwareBufferSizeBytes = 64 * 1024 * 1024;
constexpr uint32_t kMaxSamplesPerDecode = 1024;

/*
 * ========================== CUPTI PM SAMPLING API ==========================
 *
 * CUPTI PM sampling periodically collects device metrics and exports completed
 * sample intervals as CuptiPMSample values.
 *
 * The lifecycle here is:
 *
 * 1. [Configure]
 *    Builds the metric configuration, calls cuptiPmSamplingEnable(),
 *    initializes the data image (counterDataImage_). The data image is a
 *    caller-owned byte buffer initialized in CUPTI's format.
 *
 * 2. [Start]
 *    Starts hardware counter collection. CUPTI starts writing raw counter
 *    records into its hardware buffer.
 *
 * 3. [Decode] (PERIODICALLY, UNTIL STOP)
 *    Transfers records from the CUPTI's hardware buffer into the data image.
 *    Each completed sample in the data image is translated from CUPTI's format
 *    into a CuptiPMSample and pushed to the output vector. If records remain in
 *    the hardware buffer (COUNTER_DATA_FULL), we reset the image and return
 * false.
 *
 * 4. [Stop]
 *    Instruct CUPTI to stop producing records.
 *
 * 5. [Decode]
 *    Run one more decode to empty the hardware buffer.
 *
 * 6. [Disable]
 *    Disable the sampler, clear all objects.
 *
 * ===========================================================================
 */

void checkCupti(CUptiResult status, const char* call) {
  if (status == CUPTI_SUCCESS) {
    return;
  }

  const char* error = nullptr;
  cuptiGetResultString(status, &error);
  throw std::runtime_error(
      std::string{call} +
      " failed: " + (error != nullptr ? error : "unknown CUPTI error"));
}

#define CUPTI_PM_CALL(call)    \
  do {                         \
    checkCupti((call), #call); \
  } while (false)

} // namespace

CuptiPMSamplingApi::~CuptiPMSamplingApi() {
  disable();
}

void CuptiPMSamplingApi::configure(const CuptiPMSamplingConfig& config) {
  if (samplingObject_ != nullptr || hostObject_ != nullptr) {
    throw std::runtime_error(
        "Cannot configure CUPTI PM sampling after cleanup failed");
  }
  config_ = config;

  // CUPTI expects metric names as std::vector<const char*>
  metricNamePtrs_.clear();
  metricNamePtrs_.reserve(config_.metricNames.size());
  for (const auto& metricName : config_.metricNames) {
    metricNamePtrs_.push_back(metricName.c_str());
  }
  configureCupti();
}

void CuptiPMSamplingApi::configureCupti() {
  // CUpti_Device_GetChipName_Params stores device info used in other CUPTI
  // calls.
  CUpti_Device_GetChipName_Params chip{
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  chip.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_PM_CALL(cuptiDeviceGetChipName(&chip));
  if (chip.pChipName == nullptr) {
    throw std::runtime_error("cuptiDeviceGetChipName returned no chip name");
  }

  // Building the availability image. This is a CUPTI byte buffer describing
  // which raw hardware counters are available on this machine. The first call
  // to cuptiPmSamplingGetCounterAvailability is to obtain
  // counterAvailabilityImageSize, the number of bytes we need to allocate for
  // counterAvailabilityImage. Then the second call populates
  // counterAvailabilityImage.
  CUpti_PmSampling_GetCounterAvailability_Params availability{
      CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
  availability.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_PM_CALL(cuptiPmSamplingGetCounterAvailability(&availability));

  std::vector<uint8_t> counterAvailabilityImage(
      availability.counterAvailabilityImageSize);
  availability.pCounterAvailabilityImage = counterAvailabilityImage.data();
  CUPTI_PM_CALL(cuptiPmSamplingGetCounterAvailability(&availability));

  // CUpti_Profiler_Host_Initialize_Params is the evaluator which translates
  // from requested metric names into the raw counters the GPU should collect.
  // Later on, cuptiProfilerHostEvaluateToGpuValues() uses the initialized
  // hostObject_ to convert raw hardware counters into the requested metric
  // values using chip-specific formulas.
  CUpti_Profiler_Host_Initialize_Params host{
      CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  host.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
  host.pChipName = chip.pChipName;
  host.pCounterAvailabilityImage = counterAvailabilityImage.data();
  CUPTI_PM_CALL(cuptiProfilerHostInitialize(&host));
  hostObject_ = host.pHostObject;

  // Translating the requested metric strings and passing them to CUPTI.
  // This updates hostObject_ with the high-level metric names inplace.
  CUpti_Profiler_Host_ConfigAddMetrics_Params addMetrics{
      CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  addMetrics.pHostObject = hostObject_;
  addMetrics.ppMetricNames = metricNamePtrs_.data();
  addMetrics.numMetrics = metricNamePtrs_.size();
  CUPTI_PM_CALL(cuptiProfilerHostConfigAddMetrics(&addMetrics));

  // Allocation/configuration for configImage. This describes
  // the raw (low-level) counters the sampler needs to collect for the requested
  // metrics and how to schedule them.
  CUpti_Profiler_Host_GetConfigImageSize_Params size{
      CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  size.pHostObject = hostObject_;
  CUPTI_PM_CALL(cuptiProfilerHostGetConfigImageSize(&size));

  std::vector<uint8_t> configImage(size.configImageSize);
  CUpti_Profiler_Host_GetConfigImage_Params image{
      CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  image.pHostObject = hostObject_;
  image.configImageSize = configImage.size();
  image.pConfigImage = configImage.data();
  CUPTI_PM_CALL(cuptiProfilerHostGetConfigImage(&image));

  // Asking CUPTI how many passes it will take to fetch the requested metrics.
  // We do not want to make multiple passes, so reject the config if this is the
  // case.
  CUpti_Profiler_Host_GetNumOfPasses_Params passes{
      CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
  passes.configImageSize = configImage.size();
  passes.pConfigImage = configImage.data();
  CUPTI_PM_CALL(cuptiProfilerHostGetNumOfPasses(&passes));
  if (passes.numOfPasses != 1) {
    throw std::runtime_error(
        "CUPTI PM sampling requires one pass; configuration requires " +
        std::to_string(passes.numOfPasses));
  }

  // Enabling PM sampling and setting the config
  CUpti_PmSampling_Enable_Params enable{
      CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
  enable.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_PM_CALL(cuptiPmSamplingEnable(&enable));
  samplingObject_ = enable.pPmSamplingObject;

  CUpti_PmSampling_SetConfig_Params setConfig{
      CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
  setConfig.pPmSamplingObject = samplingObject_;
  setConfig.configSize = configImage.size();
  setConfig.pConfig = configImage.data();
  setConfig.hardwareBufferSize = kHardwareBufferSizeBytes;
  setConfig.samplingInterval =
      static_cast<uint64_t>(config_.samplingInterval.count());
  setConfig.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
  CUPTI_PM_CALL(cuptiPmSamplingSetConfig(&setConfig));

  // Asking CUPTI how large the (counter) data image should be.
  // Since the data image has an opage CUPTI-defined layout, the size
  // depends on sampling config, metrics, etc.
  CUpti_PmSampling_GetCounterDataSize_Params counterDataSize{
      CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
  counterDataSize.pPmSamplingObject = samplingObject_;
  counterDataSize.pMetricNames = metricNamePtrs_.data();
  counterDataSize.numMetrics = metricNamePtrs_.size();
  counterDataSize.maxSamples = kMaxSamplesPerDecode;
  CUPTI_PM_CALL(cuptiPmSamplingGetCounterDataSize(&counterDataSize));

  counterDataImage_.resize(counterDataSize.counterDataSize);
  resetImage();
}

void CuptiPMSamplingApi::resetImage() {
  CUpti_PmSampling_CounterDataImage_Initialize_Params params{
      CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  params.counterDataSize = counterDataImage_.size();
  params.pCounterData = counterDataImage_.data();
  CUPTI_PM_CALL(cuptiPmSamplingCounterDataImageInitialize(&params));
}

void CuptiPMSamplingApi::start() {
  CUpti_PmSampling_Start_Params params{
      CUpti_PmSampling_Start_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  CUPTI_PM_CALL(cuptiPmSamplingStart(&params));
}

CuptiPMSample CuptiPMSamplingApi::decodeSample(size_t sampleIndex) {
  // Decode a single CUPTI PM sample into CuptiPMSample via
  // cuptiProfilerHostEvaluateToGpuValues.
  CUpti_PmSampling_CounterData_GetSampleInfo_Params info{
      CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
  info.pPmSamplingObject = samplingObject_;
  info.pCounterDataImage = counterDataImage_.data();
  info.counterDataImageSize = counterDataImage_.size();
  info.sampleIndex = sampleIndex;
  CUPTI_PM_CALL(cuptiPmSamplingCounterDataGetSampleInfo(&info));

  CuptiPMSample sample{
      info.startTimestamp,
      info.endTimestamp,
      std::vector<double>(metricNamePtrs_.size())};
  CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate{
      CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
  evaluate.pHostObject = hostObject_;
  evaluate.pCounterDataImage = counterDataImage_.data();
  evaluate.counterDataImageSize = counterDataImage_.size();
  evaluate.rangeIndex = sampleIndex;
  evaluate.ppMetricNames = metricNamePtrs_.data();
  evaluate.numMetrics = metricNamePtrs_.size();
  evaluate.pMetricValues = sample.values.data();
  CUPTI_PM_CALL(cuptiProfilerHostEvaluateToGpuValues(&evaluate));
  return sample;
}

bool CuptiPMSamplingApi::decode(std::vector<CuptiPMSample>& samples) {
  // Fetch the next result from CUPTI. Decodes the collected counters
  // inline into the counterDataImage_.
  CUpti_PmSampling_DecodeData_Params decode{
      CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
  decode.pPmSamplingObject = samplingObject_;
  decode.pCounterDataImage = counterDataImage_.data();
  decode.counterDataImageSize = counterDataImage_.size();
  CUPTI_PM_CALL(cuptiPmSamplingDecodeData(&decode));
  if (decode.overflow != 0) {
    LOG_FIRST_N(WARNING, 1)
        << "CUPTI PM sampling hardware buffer overflowed; samples were lost";
  }

  const auto reason = decode.decodeStopReason;
  bool isBufferDrained;
  if (reason == CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS) {
    isBufferDrained = true;
  } else if (reason == CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL) {
    isBufferDrained = false;
  } else {
    throw std::runtime_error(
        "cuptiPmSamplingDecodeData returned unexpected stop reason " +
        std::to_string(static_cast<int>(reason)));
  }

  CUpti_PmSampling_GetCounterDataInfo_Params info{
      CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
  info.pCounterDataImage = counterDataImage_.data();
  info.counterDataImageSize = counterDataImage_.size();
  CUPTI_PM_CALL(cuptiPmSamplingGetCounterDataInfo(&info));

  samples.reserve(samples.size() + info.numCompletedSamples);
  for (size_t sampleIndex = 0; sampleIndex < info.numCompletedSamples;
       ++sampleIndex) {
    samples.push_back(decodeSample(sampleIndex));
  }

  resetImage();
  return isBufferDrained;
}

void CuptiPMSamplingApi::stop() {
  CUpti_PmSampling_Stop_Params params{CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  CUPTI_PM_CALL(cuptiPmSamplingStop(&params));
}

void CuptiPMSamplingApi::disable() {
  if (samplingObject_ != nullptr) {
    CUpti_PmSampling_Disable_Params params{
        CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    params.pPmSamplingObject = samplingObject_;
    if (CUPTI_CALL(cuptiPmSamplingDisable(&params)) == CUPTI_SUCCESS) {
      samplingObject_ = nullptr;
    }
  }
  if (hostObject_ != nullptr) {
    CUpti_Profiler_Host_Deinitialize_Params params{
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    params.pHostObject = hostObject_;
    if (CUPTI_CALL(cuptiProfilerHostDeinitialize(&params)) == CUPTI_SUCCESS) {
      hostObject_ = nullptr;
    }
  }

  // Retain handles on failure so cleanup can be retried
  if (samplingObject_ != nullptr || hostObject_ != nullptr) {
    return;
  }
  counterDataImage_.clear();
  config_.deviceId = -1;
  metricNamePtrs_.clear();
  config_.metricNames.clear();
  config_.samplingInterval = std::chrono::nanoseconds::zero();
}

#undef CUPTI_PM_CALL

} // namespace KINETO_NAMESPACE
