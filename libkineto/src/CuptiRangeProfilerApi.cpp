// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdio.h>
#include <stdlib.h>

#include "Logger.h"
#include "Demangle.h"

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiRangeProfilerApi.h"

#if HAS_CUPTI_PROFILER
#include <cupti.h>
#include <nvperf_host.h>
#include "cupti_call.h"
#endif // HAS_CUPTI_PROFILER

namespace KINETO_NAMESPACE {

#if HAS_CUPTI_PROFILER

/// Helper functions

// Available raw counters
std::vector<uint8_t> getCounterAvailiability(CUcontext cuContext) {
  std::vector<uint8_t> counterAvailabilityImage;

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE, nullptr};
  getCounterAvailabilityParams.ctx = cuContext;
  CUPTI_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  counterAvailabilityImage.clear();
  counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);

  getCounterAvailabilityParams.pCounterAvailabilityImage =
      counterAvailabilityImage.data();
  CUPTI_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  return counterAvailabilityImage;
}

std::string getChipName(int deviceId) {
  // Get chip name for the cuda device
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE, nullptr};

  getChipNameParams.deviceIndex = deviceId;
  CUPTI_CALL(cuptiDeviceGetChipName(&getChipNameParams));

  return getChipNameParams.pChipName;
}


// static
void CuptiRBProfilerSession::initCupti() {
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
}

// static
void CuptiRBProfilerSession::deInitCupti() {
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

// static
void CuptiRBProfilerSession::staticInit() {
  CuptiRBProfilerSession::initCupti();
}

// static
std::vector<uint8_t>& CuptiRBProfilerSession::counterAvailabilityImage() {
  static std::vector<uint8_t> counterAvailabilityImage_;
  return counterAvailabilityImage_;
}


// Setup the profiler sessions
CuptiRBProfilerSession::CuptiRBProfilerSession(
    const std::vector<std::string>& metricNames,
    int deviceId,
    int maxRanges,
    int numNestingLevels,
    CUcontext cuContext)
    : metricNames_(metricNames),
      chipName_(getChipName(deviceId)),
      deviceId_(deviceId),
      maxRanges_(maxRanges),
      numNestingLevels_(numNestingLevels),
      cuContext_(cuContext) {
  CuptiRBProfilerSession::initCupti();

  LOG(INFO) << "Initializing CUPTI profiler session : device = " << deviceId
            << " chip = " << chipName_;
  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE, nullptr};
  NVPW_CALL(NVPW_InitializeHost(&initializeHostParams));

  if (metricNames.size()) {
    if (!nvperf::getProfilerConfigImage(
            chipName_,
            metricNames,
            configImage,
            CuptiRBProfilerSession::counterAvailabilityImage().data())) {
      LOG(ERROR) << "Failed to create configImage or counterDataImagePrefix";
      return;
    }
    if (!nvperf::getCounterDataPrefixImage(
            chipName_,
            metricNames,
            counterDataImagePrefix)) {
      LOG(ERROR) << "Failed to create counterDataImagePrefix";
      return;
    }
  } else {
    LOG(ERROR) << "No metrics provided to profile";
    return;
  }

  if (!createCounterDataImage()) {
    LOG(ERROR) << "Failed to create counterDataImage";
    return;
  }

  LOG(INFO) << "Size of structs\n"
            << " config image size = " << configImage.size()  << " B"
            << " counter data image prefix = "
            << counterDataImagePrefix.size()  << " B"
            << " counter data image size = " << counterDataImage.size() / 1024
            << " KB"
            << " counter sb image size = "
            << counterDataScratchBuffer.size()  << " B";

  beginPassParams_ = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE, nullptr};
  endPassParams_ = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE, nullptr};

  initSuccess_ = true;

  // TODO profiler_map[deviceId] = this;
}

void CuptiRBProfilerSession::startInternal(
    CUpti_ProfilerRange profilerRange,
    CUpti_ProfilerReplayMode profilerReplayMode) {
  LOG(INFO) << "Starting profiler session: profiler range = "
            << ((profilerRange == CUPTI_AutoRange) ? "autorange" : "userrange")
            << " replay mode = "
            << ((profilerReplayMode == CUPTI_KernelReplay) ? "kernel" : "user");
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }

  profilerStartTs_ = std::chrono::high_resolution_clock::now();
  curRange_ = profilerRange;
  curReplay_ = profilerReplayMode;

  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE, nullptr};

  beginSessionParams.ctx = cuContext_;
  beginSessionParams.counterDataImageSize = counterDataImage.size();
  beginSessionParams.pCounterDataImage = counterDataImage.data();
  beginSessionParams.counterDataScratchBufferSize =
      counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer = counterDataScratchBuffer.data();
  beginSessionParams.range = profilerRange;
  beginSessionParams.replayMode = profilerReplayMode;
  beginSessionParams.maxRangesPerPass = maxRanges_;
  beginSessionParams.maxLaunchesPerPass = maxRanges_;

  auto status = CUPTI_CALL(cuptiProfilerBeginSession(&beginSessionParams));
  if (status != CUPTI_SUCCESS) {
    LOG(WARNING) << "Failed to start CUPTI profiler";
    initSuccess_ = false;
    return;
  }

  // Set counter configuration
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE, nullptr};

  setConfigParams.ctx = cuContext_;
  setConfigParams.pConfig = configImage.data();
  setConfigParams.configSize = configImage.size();
  setConfigParams.passIndex = 0;
  setConfigParams.minNestingLevel = 1;
  setConfigParams.numNestingLevels = numNestingLevels_;
  status = CUPTI_CALL(cuptiProfilerSetConfig(&setConfigParams));

  if (status != CUPTI_SUCCESS) {
    LOG(WARNING) << "Failed to configure CUPTI profiler";
    initSuccess_ = false;
    return;
  }
  profilerInitDoneTs_ = std::chrono::high_resolution_clock::now();
}

void CuptiRBProfilerSession::stop() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  LOG(INFO) << "Stop profiler session on device = " << deviceId_;

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerEndSession(&endSessionParams));

  profilerStopTs_ = std::chrono::high_resolution_clock::now();
}

void CuptiRBProfilerSession::beginPass() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  CUPTI_CALL(cuptiProfilerBeginPass(&beginPassParams_));
}

bool CuptiRBProfilerSession::endPass() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return true;
  }
  CUPTI_CALL(cuptiProfilerEndPass(&endPassParams_));
  return endPassParams_.allPassesSubmitted;
}

void CuptiRBProfilerSession::flushCounterData() {
  LOG(INFO) << "Flushing counter data on device = " << deviceId_;
  CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
      CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
}

/// Enable and disable the profiler
void CuptiRBProfilerSession::enable() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}

void CuptiRBProfilerSession::disable() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
}

void CuptiRBProfilerSession::asyncStartAndEnable(
    CUpti_ProfilerRange /*profilerRange*/,
    CUpti_ProfilerReplayMode /*profilerReplayMode*/) {
  /* TBD */
}

void CuptiRBProfilerSession::asyncDisableAndStop() {
  /* TBD */
}

/// User range based profiling
void CuptiRBProfilerSession::pushRange(const std::string& rangeName) {
  LOG(INFO) << " CUPTI pushrange ( " << rangeName << " )";
  CUpti_Profiler_PushRange_Params pushRangeParams = {
      CUpti_Profiler_PushRange_Params_STRUCT_SIZE, nullptr};
  pushRangeParams.pRangeName = rangeName.c_str();
  CUPTI_CALL(cuptiProfilerPushRange(&pushRangeParams));
}

void CuptiRBProfilerSession::popRange() {
  LOG(INFO) << "Pop User range";
  CUpti_Profiler_PopRange_Params popRangeParams = {
      CUpti_Profiler_PopRange_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerPopRange(&popRangeParams));
}

CuptiProfilerResult CuptiRBProfilerSession::evalualteMetrics(
    bool verbose) {
  if (!initSuccess_) {
    LOG(WARNING) << "Profiling failed, no results to return";
    return {};
  }

  LOG(INFO) << "Total kernels logged = " << kernelNames_.size();
  if (verbose) {
    for (const auto& kernel : kernelNames_) {
      std::cout << demangle(kernel) << std::endl;
    }
    LOG(INFO) << "Profiler Range data : ";
  }
  LOG(INFO) << "Profiler Range data : ";

  auto results = nvperf::evalMetricValues(
      chipName_, counterDataImage, metricNames_, verbose /*verbose*/);

  // profiler end-end duration
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      profilerStopTs_ - profilerStartTs_);

  auto init_dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      profilerInitDoneTs_ - profilerStartTs_);
  LOG(INFO) << "Total profiler time = " << duration_ms.count() << " ms";
  LOG(INFO) << "Total profiler init time = " << init_dur_ms.count() << " ms";

  return results;
}

void CuptiRBProfilerSession::saveCounterData(
    const std::string& /*CounterDataFileName*/,
    const std::string& /*CounterDataSBFileName*/) {
  /* TBD write binary files for counter data and counter scratch buffer */
}

/// Setup counter data
bool CuptiRBProfilerSession::createCounterDataImage() {
  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = counterDataImagePrefix.data();
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = maxRanges_;
  counterDataImageOptions.maxNumRangeTreeNodes = maxRanges_;
  counterDataImageOptions.maxRangeNameLength = 64;

  // Calculate size of counter data image
  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE, nullptr};
  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
  counterDataImage.resize(calculateSizeParams.counterDataImageSize);

  // Initialize counter data image
  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
    CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE, nullptr};
  initializeParams.sizeofCounterDataImageOptions =
    CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
    calculateSizeParams.counterDataImageSize;
  initializeParams.pCounterDataImage = counterDataImage.data();
  CUPTI_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  // Calculate counter Scratch Buffer size
  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
    scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE, nullptr};

  scratchBufferSizeParams.counterDataImageSize =
    calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
    initializeParams.pCounterDataImage;
  CUPTI_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
    &scratchBufferSizeParams));

  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  // Initialize scratch buffer
  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
    initScratchBufferParams = {
      CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE, nullptr};

  initScratchBufferParams.counterDataImageSize =
    calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage =
    initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
    scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
    counterDataScratchBuffer.data();

  CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));

  return true;
}

#elif defined(HAS_CUPTI)

// Create empty stubs for the API when CUPTI is not present.
CuptiRBProfilerSession::CuptiRBProfilerSession(
    const std::vector<std::string>& metricNames,
    int deviceId,
    int maxRanges,
    int numNestingLevels,
    CUcontext cuContext)
    : metricNames_(metricNames),
      deviceId_(deviceId),
      maxRanges_(maxRanges),
      numNestingLevels_(numNestingLevels),
      cuContext_(cuContext) {}
void CuptiRBProfilerSession::stop() {}
void CuptiRBProfilerSession::enable() {}
void CuptiRBProfilerSession::disable() {}
void CuptiRBProfilerSession::beginPass() {}
bool CuptiRBProfilerSession::endPass() { return true; }
void CuptiRBProfilerSession::flushCounterData() {}
void CuptiRBProfilerSession::pushRange(const std::string& /*rangeName*/) {}
void CuptiRBProfilerSession::popRange() {}
void CuptiRBProfilerSession::asyncStartAndEnable(
    CUpti_ProfilerRange /*profilerRange*/,
    CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}
void CuptiRBProfilerSession::asyncDisableAndStop() {}
CuptiProfilerResult CuptiRBProfilerSession::evalualteMetrics(bool verbose) {
  static CuptiProfilerResult res;
  return res;
};
void CuptiRBProfilerSession::saveCounterData(
    const std::string& /*CounterDataFileName*/,
    const std::string& /*CounterDataSBFileName*/) {}
void CuptiRBProfilerSession::initCupti() {}
void CuptiRBProfilerSession::deInitCupti() {}
void CuptiRBProfilerSession::staticInit() {}
bool CuptiRBProfilerSession::createCounterDataImage() { return true; }
void CuptiRBProfilerSession::startInternal(
    CUpti_ProfilerRange /*profilerRange*/,
    CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}
std::vector<uint8_t>& CuptiRBProfilerSession::counterAvailabilityImage() {
  static std::vector<uint8_t> _vec;
  return _vec;
}
#endif // HAS_CUPTI_PROFILER

} // namespace KINETO_NAMESPACE
