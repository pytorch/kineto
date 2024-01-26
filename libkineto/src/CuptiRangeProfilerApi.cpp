/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef HAS_CUPTI
#include <cupti.h>
#include <nvperf_host.h>
#endif // HAS_CUPTI
#include <mutex>
#include <unordered_map>

#ifdef HAS_CUPTI
#include "cupti_call.h"
#endif

#include "time_since_epoch.h"
#include "Logger.h"
#include "Demangle.h"

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiRangeProfilerApi.h"

#if HAS_CUPTI_RANGE_PROFILER
#include <cupti.h>
#include <nvperf_host.h>
#include "cupti_call.h"
#endif // HAS_CUPTI_RANGE_PROFILER

namespace KINETO_NAMESPACE {

TraceSpan CuptiRBProfilerSession::getProfilerTraceSpan() {
  return TraceSpan(
      timeSinceEpoch(profilerStartTs_),
      timeSinceEpoch(profilerStopTs_),
      "__cupti_profiler__"
  );
}

#if HAS_CUPTI_RANGE_PROFILER
constexpr char kRootUserRangeName[] = "__profile__";
constexpr int kCallbacksCountToFlush = 500;

// Shared state to track one Cupti Profiler API per Device
namespace {
// per device profiler maps
std::unordered_map<uint32_t, CuptiRBProfilerSession*> profiler_map;
std::unordered_map<uint32_t, bool> enable_flag;
std::unordered_map<uint32_t, bool> disable_flag;

std::mutex contextMutex_;
std::unordered_map<CUcontext, int> ctx_to_dev;
std::set<uint32_t> active_devices;
}

// forward declarations
void __trackCudaCtx(CUcontext ctx, uint32_t device_id, CUpti_CallbackId cbid);
void __trackCudaKernelLaunch(CUcontext ctx, const char* kernelName);

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

inline uint32_t getDevID(CUcontext ctx) {
  uint32_t device_id = UINT32_MAX;
  CUPTI_CALL(cuptiGetDeviceId(ctx, &device_id));
  if (device_id == UINT32_MAX) {
    LOG(ERROR) << "Could not determine dev id for = " << ctx;
  }
  return device_id;
}

// We use CUPTI Callback functions in three ways :
//   1. Track cuda contexts and maintain a list of active GPUs to profile
//   2. Callbacks on kernel launches to track the name of automatic
//      ranges that correspond to names of kernels
//   3. Lastly CUPTI range profiler has to be enabled on the same thread executing
//      the CUDA kernels. We use Callbacks to enable the profiler
//      asynchronously from another thread.

void disableKernelCallbacks();

void trackCudaCtx(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  auto *d = reinterpret_cast<const CUpti_ResourceData*>(cbInfo);
  auto ctx = d->context;
  uint32_t device_id = getDevID(ctx);

  if (device_id == UINT32_MAX) {
    return;
  }

  __trackCudaCtx(ctx, device_id, cbid);
}

void __trackCudaCtx(CUcontext ctx, uint32_t device_id, CUpti_CallbackId cbid) {
  std::lock_guard<std::mutex> g(contextMutex_);
  if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
    VLOG(0) << "CUPTI Profiler observed CUDA Context created = "
            << ctx << " device id = " << device_id;
    active_devices.insert(device_id);
    ctx_to_dev[ctx] = device_id;

  } else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    VLOG(0) << "CUPTI Profiler observed CUDA Context destroyed = "
            << ctx << " device id = " << device_id;
    auto it = active_devices.find(device_id);
    if (it != active_devices.end()) {
      active_devices.erase(it);
      ctx_to_dev.erase(ctx);
    }
  }
}

void trackCudaKernelLaunch(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* cbInfo) {
  VLOG(1) << " Trace : Callback name = "
          << (cbInfo->symbolName ?  cbInfo->symbolName: "")
          << " context ptr = " << cbInfo->context;
  auto ctx = cbInfo->context;
  // should be in CUPTI_API_ENTER call site
  if (cbInfo->callbackSite != CUPTI_API_ENTER) {
    return;
  }
  __trackCudaKernelLaunch(ctx, cbInfo->symbolName);
}

void __trackCudaKernelLaunch(
    CUcontext ctx,
    const char* kernelName) {
  VLOG(0) << " Tracking kernel name = " << (kernelName ? kernelName : "")
          << " context ptr = " << ctx;

  uint32_t device_id = 0;
  auto it = ctx_to_dev.find(ctx);
  if (it == ctx_to_dev.end()) {
    // Warning here could be too noisy
    VLOG(0) << " Could not find corresponding device to ctx = " << ctx;
    return;
  } else {
    device_id = it->second;
  }

  auto pit = profiler_map.find(device_id);
  if (pit == profiler_map.end() || pit->second == nullptr) {
    return;
  }
  auto profiler = pit->second;

  if (enable_flag[device_id]) {
    LOG(INFO) << "Callback handler is enabling cupti profiler";
    profiler->startAndEnable();
    enable_flag[device_id] = false;

  } else if (disable_flag[device_id]) {
    LOG(INFO) << "Callback handler is disabling cupti profiler";
    profiler->disableAndStop();
    return;
  }

  if (profiler->curRange_ == CUPTI_AutoRange) {
    profiler->logKernelName(kernelName ? kernelName : "__missing__");
  }

  /* TODO add per kernel time logging
  if (measure_per_kernel) {
    profiler->kernelStartTs_.push_back(
        std::chrono::high_resolution_clock::now());
  }
  */

  // periodically flush profiler data from GPU
  if (profiler->numCallbacks_ % kCallbacksCountToFlush == 0) {
    profiler->flushCounterData();
  }
  profiler->numCallbacks_++;
}

void enableKernelCallbacks() {
  auto cbapi = CuptiCallbackApi::singleton();
  bool status = cbapi->enableCallback(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  if (!status) {
    LOG(WARNING) << "CUPTI Range Profiler unable to "
                 << "enable cuda kernel launch callback";
    return;
  }
  LOG(INFO) << "CUPTI Profiler kernel callbacks enabled";
}

void disableKernelCallbacks() {
  auto cbapi = CuptiCallbackApi::singleton();
  bool status = cbapi->disableCallback(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  if (!status) {
    LOG(WARNING) << "CUPTI Range Profiler unable to "
                 << "disable cuda kernel launch callback";
    return;
  }
  LOG(INFO) << "CUPTI Profiler kernel callbacks disabled";
}

// static
std::set<uint32_t> CuptiRBProfilerSession::getActiveDevices() {
  std::lock_guard<std::mutex> g(contextMutex_);
  return active_devices;
}

// static
bool CuptiRBProfilerSession::initCupti() {
  // This call will try to load the libnvperf_host.so library and is known
  // to break address sanitizer based flows. Only call this init
  // when you plan to use CUPTI range profiler
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE, nullptr};
  CUptiResult status = CUPTI_CALL_NOWARN(
      cuptiProfilerInitialize(&profilerInitializeParams));
  return (status == CUPTI_SUCCESS);
}

// static
void CuptiRBProfilerSession::deInitCupti() {
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE, nullptr};
  CUPTI_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

// static
bool CuptiRBProfilerSession::staticInit() {
  // Register CUPTI callbacks
  auto cbapi = CuptiCallbackApi::singleton();
  CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RESOURCE;
  bool status = cbapi->registerCallback(
      domain, CuptiCallbackApi::RESOURCE_CONTEXT_CREATED, trackCudaCtx);
  status = status && cbapi->registerCallback(
      domain, CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED, trackCudaCtx);
  status = status && cbapi->enableCallback(
      domain, CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
  status = status && cbapi->enableCallback(
      domain, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);

  if (!status) {
    LOG(WARNING) << "CUPTI Range Profiler unable to attach cuda context "
                 << "create and destroy callbacks";
    CUPTI_CALL(cbapi->getCuptiStatus());
    return false;
  }

  domain = CUPTI_CB_DOMAIN_RUNTIME_API;
  status = cbapi->registerCallback(
      domain, CuptiCallbackApi::CUDA_LAUNCH_KERNEL, trackCudaKernelLaunch);

  if (!status) {
    LOG(WARNING) << "CUPTI Range Profiler unable to attach cuda kernel "
                 << "launch callback";
    return false;
  }

  return true;
}

// static
std::vector<uint8_t>& CuptiRBProfilerSession::counterAvailabilityImage() {
  static std::vector<uint8_t> counterAvailabilityImage_;
  return counterAvailabilityImage_;
}


// Setup the profiler sessions
CuptiRBProfilerSession::CuptiRBProfilerSession(
    const CuptiRangeProfilerOptions& opts)
    : metricNames_(opts.metricNames),
      deviceId_(opts.deviceId),
      maxRanges_(opts.maxRanges),
      numNestingLevels_(opts.numNestingLevels),
      cuContext_(opts.cuContext) {
  // used in unittests only
  if (opts.unitTest) {
    initSuccess_ = true;
    profiler_map[deviceId_] = this;
    return;
  }

  chipName_ = getChipName(opts.deviceId);

  if (!CuptiRBProfilerSession::initCupti()) {
    LOG(ERROR) << "Failed to initialize CUPTI range profiler.";
    return;
  }

  LOG(INFO) << "Initializing CUPTI range profiler session : device = " << deviceId_
            << " chip = " << chipName_;
  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE, nullptr};
  NVPW_CALL(NVPW_InitializeHost(&initializeHostParams));

  if (metricNames_.size()) {
    if (!nvperf::getProfilerConfigImage(
            chipName_,
            metricNames_,
            configImage,
            CuptiRBProfilerSession::counterAvailabilityImage().data())) {
      LOG(ERROR) << "Failed to create configImage or counterDataImagePrefix";
      return;
    }
    if (!nvperf::getCounterDataPrefixImage(
            chipName_,
            metricNames_,
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

  LOG(INFO) << "Size of structs"
            << " config image size = " << configImage.size()  << " B"
            << " counter data image prefix = "
            << counterDataImagePrefix.size()  << " B"
            << " counter data image size = " << counterDataImage.size() / 1024
            << " KB"
            << " counter sb image size = "
            << counterDataScratchBuffer.size()  << " B";

  beginPassParams_ = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE, nullptr};
  beginPassParams_.ctx = cuContext_;
  endPassParams_ = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE, nullptr};
  endPassParams_.ctx = cuContext_;

  initSuccess_ = true;
  profiler_map[deviceId_] = this;
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

  if (cuContext_ == nullptr) {
    for (const auto& it : ctx_to_dev) {
      if (it.second == deviceId_) {
        cuContext_ = it.first;
        break;
      }
    }
    LOG(INFO) << " Cupti Profiler using CUDA context = " << cuContext_;
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
    LOG(WARNING) << "Failed to start CUPTI range profiler";
    initSuccess_ = false;
    return;
  }

  // Set counter configuration
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE, nullptr};

  setConfigParams.ctx = cuContext_;
  setConfigParams.pConfig = configImage.data();
  setConfigParams.configSize = configImage.size();
  setConfigParams.minNestingLevel = 1;
  setConfigParams.numNestingLevels = numNestingLevels_;
  setConfigParams.passIndex = 0;
  setConfigParams.targetNestingLevel = setConfigParams.minNestingLevel;
  status = CUPTI_CALL(cuptiProfilerSetConfig(&setConfigParams));

  if (status != CUPTI_SUCCESS) {
    LOG(WARNING) << "Failed to configure CUPTI range profiler";
    initSuccess_ = false;
    return;
  }
  profilerInitDoneTs_ = std::chrono::high_resolution_clock::now();

  if (curRange_ == CUPTI_AutoRange) {
    enableKernelCallbacks();
  }
  profilingActive_ = true;
}

void CuptiRBProfilerSession::stop() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  LOG(INFO) << "Stop profiler session on device = " << deviceId_;

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE, nullptr};
  unsetConfigParams.ctx = cuContext_;
  CUPTI_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE, nullptr};
  endSessionParams.ctx = cuContext_;
  CUPTI_CALL(cuptiProfilerEndSession(&endSessionParams));

  disableKernelCallbacks();

  profilerStopTs_ = std::chrono::high_resolution_clock::now();
  profilingActive_ = false;
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
  flushCounterDataParams.ctx = cuContext_;
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
  enableProfilingParams.ctx = cuContext_;
  CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}

void CuptiRBProfilerSession::disable() {
  if (!initSuccess_) {
    LOG(WARNING) << __func__ << "() bailing out since initialization failed";
    return;
  }
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE, nullptr};
  disableProfilingParams.ctx = cuContext_;
  CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
}

/// User range based profiling
void CuptiRBProfilerSession::pushRange(const std::string& rangeName) {
  LOG(INFO) << " CUPTI pushrange ( " << rangeName << " )";
  CUpti_Profiler_PushRange_Params pushRangeParams = {
      CUpti_Profiler_PushRange_Params_STRUCT_SIZE, nullptr};
  pushRangeParams.ctx = cuContext_;
  pushRangeParams.pRangeName = rangeName.c_str();
  CUPTI_CALL(cuptiProfilerPushRange(&pushRangeParams));
}

void CuptiRBProfilerSession::popRange() {
  LOG(INFO) << " CUPTI pop range";
  CUpti_Profiler_PopRange_Params popRangeParams = {
      CUpti_Profiler_PopRange_Params_STRUCT_SIZE, nullptr};
  popRangeParams.ctx = cuContext_;
  CUPTI_CALL(cuptiProfilerPopRange(&popRangeParams));
}

void CuptiRBProfilerSession::startAndEnable() {
  startInternal(curRange_, curReplay_);
  if (curReplay_ == CUPTI_UserReplay) {
    beginPass();
  }
  enable();
  if (curRange_ == CUPTI_UserRange) {
    pushRange(kRootUserRangeName);
  }
  enable_flag[deviceId_] = false;
}

void CuptiRBProfilerSession::disableAndStop() {
  if (curRange_ == CUPTI_UserRange) {
    popRange();
  }
  disable();
  if (curReplay_ == CUPTI_UserReplay) {
    endPass();
    flushCounterData();
  }
  stop();
  disable_flag[deviceId_] = false;
}

void CuptiRBProfilerSession::asyncStartAndEnable(
    CUpti_ProfilerRange profilerRange,
    CUpti_ProfilerReplayMode profilerReplayMode) {
  LOG(INFO) << "Starting CUPTI range profiler asynchronously on device = "
            << deviceId_ << " profiler range = "
            << ((profilerRange == CUPTI_AutoRange) ? "autorange" : "userrange")
            << " replay mode = "
            << ((profilerReplayMode == CUPTI_KernelReplay) ? "kernel" : "user");
  curReplay_ = profilerReplayMode;
  curRange_ = profilerRange;
  enable_flag[deviceId_] = true;
  enableKernelCallbacks();
}

void CuptiRBProfilerSession::asyncDisableAndStop() {
  LOG(INFO) << "Stopping CUPTI range profiler asynchronously on device = "
            << deviceId_ << " cu context = " << cuContext_;
  disable_flag[deviceId_] = true;
}


CuptiProfilerResult CuptiRBProfilerSession::evaluateMetrics(
    bool verbose) {
  if (!initSuccess_) {
    LOG(WARNING) << "Profiling failed, no results to return";
    return {};
  }
  if (profilingActive_) {
    disableAndStop();
  }

  LOG(INFO) << "Total kernels logged = " << kernelNames_.size();
  if (verbose) {
    for (const auto& kernel : kernelNames_) {
      std::cout << demangle(kernel) << std::endl;
    }
    LOG(INFO) << "Profiler Range data : ";
  }

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

  initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
    scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
    counterDataScratchBuffer.data();

  CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));

  return true;
}

CuptiRBProfilerSession::~CuptiRBProfilerSession() {
  if (initSuccess_) {
    CuptiRBProfilerSession::deInitCupti();
  }
}

#else

// Create empty stubs for the API when CUPTI is not present.
CuptiRBProfilerSession::CuptiRBProfilerSession(
    const CuptiRangeProfilerOptions& opts)
    : metricNames_(opts.metricNames),
      deviceId_(opts.deviceId),
      maxRanges_(opts.maxRanges),
      numNestingLevels_(opts.numNestingLevels),
      cuContext_(opts.cuContext) {};
CuptiRBProfilerSession::~CuptiRBProfilerSession() {}
void CuptiRBProfilerSession::stop() {}
void CuptiRBProfilerSession::enable() {}
void CuptiRBProfilerSession::disable() {}
void CuptiRBProfilerSession::beginPass() {}
bool CuptiRBProfilerSession::endPass() { return true; }
void CuptiRBProfilerSession::flushCounterData() {}
void CuptiRBProfilerSession::pushRange(const std::string& /*rangeName*/) {}
void CuptiRBProfilerSession::popRange() {}
void CuptiRBProfilerSession::startAndEnable() {}
void CuptiRBProfilerSession::disableAndStop() {}
void CuptiRBProfilerSession::asyncStartAndEnable(
    CUpti_ProfilerRange /*profilerRange*/,
    CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}
void CuptiRBProfilerSession::asyncDisableAndStop() {}
CuptiProfilerResult CuptiRBProfilerSession::evaluateMetrics(bool verbose) {
  static CuptiProfilerResult res;
  return res;
};
void CuptiRBProfilerSession::saveCounterData(
    const std::string& /*CounterDataFileName*/,
    const std::string& /*CounterDataSBFileName*/) {}
bool CuptiRBProfilerSession::initCupti() { return false; }
void CuptiRBProfilerSession::deInitCupti() {}
bool CuptiRBProfilerSession::staticInit() { return false; }
std::set<uint32_t> CuptiRBProfilerSession::getActiveDevices() { return {}; }
bool CuptiRBProfilerSession::createCounterDataImage() { return true; }
void CuptiRBProfilerSession::startInternal(
    CUpti_ProfilerRange /*profilerRange*/,
    CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}
std::vector<uint8_t>& CuptiRBProfilerSession::counterAvailabilityImage() {
  static std::vector<uint8_t> _vec;
  return _vec;
}
#endif // HAS_CUPTI_RANGE_PROFILER

std::unique_ptr<CuptiRBProfilerSession>
CuptiRBProfilerSessionFactory::make(const CuptiRangeProfilerOptions& opts) {
  return std::make_unique<CuptiRBProfilerSession>(opts);
}

namespace testing {

void trackCudaCtx(CUcontext ctx, uint32_t device_id, CUpti_CallbackId cbid) {
#if HAS_CUPTI_RANGE_PROFILER
  __trackCudaCtx(ctx, device_id, cbid);
#endif // HAS_CUPTI_RANGE_PROFILER
}

void trackCudaKernelLaunch(CUcontext ctx, const char* kernelName) {
#if HAS_CUPTI_RANGE_PROFILER
  __trackCudaKernelLaunch(ctx, kernelName);
#endif // HAS_CUPTI_RANGE_PROFILER
}

} // namespace testing
} // namespace KINETO_NAMESPACE
