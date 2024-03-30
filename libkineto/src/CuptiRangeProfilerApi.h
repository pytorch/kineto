/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef HAS_CUPTI
#include <cuda.h>
#include <cuda_runtime_api.h>
// Using CUDA 11 and above due to usage of API: cuptiProfilerGetCounterAvailability.
#if defined(USE_CUPTI_RANGE_PROFILER) && defined(CUDART_VERSION) && CUDART_VERSION >= 10000 && CUDA_VERSION >= 11000
#define HAS_CUPTI_RANGE_PROFILER 1
#endif // CUDART_VERSION > 10.00 and CUDA_VERSION >= 11.00
#endif // HAS_CUPTI

#if HAS_CUPTI_RANGE_PROFILER
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#else
enum CUpti_ProfilerRange
{
  CUPTI_AutoRange,
  CUPTI_UserRange,
};

enum CUpti_ProfilerReplayMode
{
  CUPTI_KernelReplay,
  CUPTI_UserReplay,
};
#endif // HAS_CUPTI_RANGE_PROFILER

#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <set>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "TraceSpan.h"
#include "CuptiCallbackApi.h"
#include "CuptiNvPerfMetric.h"

/* Cupti Range based profiler session
 * See : https://docs.nvidia.com/cupti/Cupti/r_main.html#r_profiler
 */

namespace KINETO_NAMESPACE {

// Initialize and configure CUPTI Profiler counters.
// - Metric names must be provided as string vector.
// - Supported values by CUPTI can be found at -
//   https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_metrics_api
struct CuptiRangeProfilerOptions {
  std::vector<std::string> metricNames;
  int deviceId = 0;
  int maxRanges = 1;
  int numNestingLevels = 1;
  CUcontext cuContext = nullptr;
  bool unitTest = false;
};

class CuptiRBProfilerSession {
 public:

  explicit CuptiRBProfilerSession(const CuptiRangeProfilerOptions& opts);

  virtual ~CuptiRBProfilerSession();

  // Start profiling session
  // This function has to be called from the CPU thread running
  // the CUDA context. If this is not the case asyncStartAndEnable()
  // can be used
  void start(
      CUpti_ProfilerRange profilerRange = CUPTI_AutoRange,
      CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay) {
    startInternal(profilerRange, profilerReplayMode);
  }

  // Stop profiling session
  virtual void stop();

  virtual void enable();
  virtual void disable();

  // Profiler passes
  //  GPU hardware has limited performance monitoring resources
  //  the CUPTI profiler may need to run multiple passes to collect
  //  data for a given range
  //  If we use kernel replay model the kernels are automatically replayed
  //  else, you can use the beginPass() and endPass() functions below
  //  for user to manage the replays

  // starts a profiler pass with given kernels in between
  virtual void beginPass();

  // end a profiler pass with given kernels in between
  // returns true if no more passes are required
  virtual bool endPass();

  // flushes the counter data - required if you use user replay
  virtual void flushCounterData();

  // Each pass can contain multiple of ranges
  //  metrics configured in a pass are collected per each range-stack.
  virtual void pushRange(const std::string& rangeName);
  virtual void popRange();

  // utilities for common operations
  void startAndEnable();
  void disableAndStop();

  // Async APIs : these will can be called from another thread
  // outside the CUDA context being profiled
  void asyncStartAndEnable(
      CUpti_ProfilerRange profilerRange = CUPTI_AutoRange,
      CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay);
  void asyncDisableAndStop();

  void printMetrics() {
    evaluateMetrics(true);
  }

 TraceSpan getProfilerTraceSpan();

  virtual CuptiProfilerResult evaluateMetrics(bool verbose = false);

  void saveCounterData(
      const std::string& CounterDataFileName,
      const std::string& CounterDataSBFileName);

  // This is not thread safe so please only call after
  // profiling has stopped
  const std::vector<std::string>& getKernelNames() const {
    return kernelNames_;
  }

  int deviceId() const {
    return deviceId_;
  }

  bool profilingActive() const {
    return profilingActive_;
  }

  static std::set<uint32_t> getActiveDevices();

  static bool initCupti();

  static void deInitCupti();

  static bool staticInit();

  static void setCounterAvailabilityImage(std::vector<uint8_t> img) {
    counterAvailabilityImage() = img;
  }

 protected:
  virtual void startInternal(
      CUpti_ProfilerRange profilerRange,
      CUpti_ProfilerReplayMode profilerReplayMode);

  CUpti_ProfilerRange curRange_ = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode curReplay_ = CUPTI_KernelReplay;

  std::chrono::time_point<std::chrono::high_resolution_clock>
    profilerStartTs_, profilerStopTs_, profilerInitDoneTs_;

 private:

  bool createCounterDataImage();

  // log kernel name that used with callbacks
  void logKernelName(const char* kernel) {
    std::lock_guard<std::mutex> lg(kernelNamesMutex_);
    kernelNames_.emplace_back(kernel);
  }

  std::vector<std::string> metricNames_;
  std::string chipName_;

  uint32_t deviceId_ = 0;
  int maxRanges_;
  int numNestingLevels_;
  CUcontext cuContext_;


  // data buffers for configuration and counter data collection
  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataScratchBuffer;

  std::mutex kernelNamesMutex_;
  // raw kernel names (not demangled)
  std::vector<std::string> kernelNames_;

  uint32_t numCallbacks_ = 0;

  static std::vector<uint8_t>& counterAvailabilityImage();

#if HAS_CUPTI_RANGE_PROFILER
  CUpti_Profiler_BeginPass_Params beginPassParams_;
  CUpti_Profiler_EndPass_Params endPassParams_;
#endif

  bool initSuccess_ = false;
  bool profilingActive_ = false;

  friend void __trackCudaKernelLaunch(CUcontext ctx, const char* kernelName);
};

// Factory class used by the wrapping CuptiProfiler object
struct ICuptiRBProfilerSessionFactory {
  virtual std::unique_ptr<CuptiRBProfilerSession> make(
      const CuptiRangeProfilerOptions& opts) = 0;
  virtual ~ICuptiRBProfilerSessionFactory() {}
};

struct CuptiRBProfilerSessionFactory : ICuptiRBProfilerSessionFactory {
  std::unique_ptr<CuptiRBProfilerSession> make(
      const CuptiRangeProfilerOptions& opts) override;
};


// called directly only in unit tests
namespace testing {

void trackCudaCtx(CUcontext ctx, uint32_t device_id, CUpti_CallbackId cbid);
void trackCudaKernelLaunch(CUcontext ctx, const char* kernelName);

} // namespace testing

} // namespace KINETO_NAMESPACE
