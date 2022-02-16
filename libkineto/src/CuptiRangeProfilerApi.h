// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(HAS_CUPTI) && CUDA_VERSION >= 10010
#if defined(CUDART_VERSION) && CUDART_VERSION > 10000 && CUDART_VERSION < 11040
#define HAS_CUPTI_PROFILER 1
#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#endif // CUDART_VERSION > 10.00 and < 11.04
#endif // HAS_CUPTI && CUDA_VERSION >= 10.10

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiNvPerfMetric.h"

/* Cupti Range based profiler session
 * See : https://docs.nvidia.com/cupti/Cupti/r_main.html#r_profiler
 */

namespace KINETO_NAMESPACE {

#if HAS_CUPTI_PROFILER
class CuptiRBProfilerSession {
 public:
  // Initialize and configure CUPTI Profiler counters.
  // - Metric names must be provided as string vector.
  // - Supported values by CUPTI can be found at -
  //   https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_metrics_api
  explicit CuptiRBProfilerSession(
      const std::vector<std::string>& metricNames,
      int deviceId,
      int maxRanges,
      int numNestingLevels = 1,
      CUcontext cuContext = nullptr);


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
  void stop();

  void enable();
  void disable();

  // Profiler passes
  //  GPU hardware has limited performance monitoring resources
  //  the CUPTI profiler may need to run multiple passes to collect
  //  data for a given range
  //  If we use kernel replay model the kernels are automatically replayed
  //  else, you can use the beginPass() and endPass() functions below
  //  for user to manage the replays

  // starts a profiler pass with given kernels in between
  void beginPass();

  // end a profiler pass with given kernels in between
  // returns true if no more passes are required
  bool endPass();

  // flushes the counter data - required if you use user replay
  void flushCounterData();

  // Each pass can contain multiple of ranges
  //  metrics configured in a pass are collected per each range-stack.
  void pushRange(const std::string& rangeName);
  void popRange();

  // Async APIs : these will can be called from another thread
  // outside the CUDA context being profiled
  void asyncStartAndEnable(
      CUpti_ProfilerRange profilerRange = CUPTI_AutoRange,
      CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay);
  void asyncDisableAndStop();

  void printMetrics() {
    evalualteMetrics(true);
  }

  CuptiProfilerResult evalualteMetrics(bool verbose = false);

  void saveCounterData(
      const std::string& CounterDataFileName,
      const std::string& CounterDataSBFileName);

  static void initCupti();

  static void deInitCupti();

  static void staticInit();

  static void setCounterAvailabilityImage(std::vector<uint8_t> img) {
    counterAvailabilityImage() = img;
  }

 private:

  bool createCounterDataImage();

  void startInternal(
      CUpti_ProfilerRange profilerRange,
      CUpti_ProfilerReplayMode profilerReplayMode);

  // log kernel name that used with callbacks
  void logKernelName(const char* kernel) {
    std::lock_guard<std::mutex> lg(kernelNamesMutex_);
    kernelNames_.emplace_back(kernel);
  }


  std::vector<std::string> metricNames_;
  std::string chipName_;

  uint32_t deviceId_;
  int maxRanges_;
  int numNestingLevels_;
  CUcontext cuContext_;

  CUpti_ProfilerRange curRange_ = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode curReplay_ = CUPTI_KernelReplay;

  // data buffers for configuration and counter data collection
  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataScratchBuffer;

  std::chrono::time_point<std::chrono::high_resolution_clock> profilerStartTs_;
  std::chrono::time_point<std::chrono::high_resolution_clock>
      profilerInitDoneTs_;
  std::chrono::time_point<std::chrono::high_resolution_clock> profilerStopTs_;

  std::mutex kernelNamesMutex_;
  // raw kernel names (not demangled)
  std::vector<std::string> kernelNames_;

  static std::vector<uint8_t>& counterAvailabilityImage();

  CUpti_Profiler_BeginPass_Params beginPassParams_;
  CUpti_Profiler_EndPass_Params endPassParams_;

  bool initSuccess_ = false;
};
#else // !HAS_CUPTI_PROFILER

// Create empty stubs for the API when CUPTI is not present.
using CUpti_ProfilerRange = enum
{
    CUPTI_Range_INVALID,
    CUPTI_AutoRange,
    CUPTI_UserRange,
    CUPTI_Range_COUNT,
};

using CUpti_ProfilerReplayMode = enum
{
    CUPTI_Replay_INVALID,
    CUPTI_ApplicationReplay,
    CUPTI_KernelReplay,
    CUPTI_UserReplay,
    CUPTI_Replay_COUNT,
};

class CuptiRBProfilerSession {
public:
  explicit CuptiRBProfilerSession(
      const std::vector<std::string>& /*metricNames*/,
      int /*deviceId*/,
      int /*maxRanges*/,
      int /*numNestingLevels*/,
      void* /*cuContext*/) {}

  void start(
      CUpti_ProfilerRange /*profilerRange*/,
      CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}

  void stop() {}
  void enable() {}
  void disable() {}
  void beginPass() {}
  bool endPass() { return true; }
  void flushCounterData() {}
  void pushRange(const std::string& /*rangeName*/) {}
  void popRange() {}
  void asyncStartAndEnable(
      CUpti_ProfilerRange /*profilerRange*/,
      CUpti_ProfilerReplayMode /*profilerReplayMode*/) {}
  void asyncDisableAndStop() {}
  void printMetrics() {}
  void saveCounterData(
      const std::string& /*CounterDataFileName*/,
      const std::string& /*CounterDataSBFileName*/) {}
  static void initCupti(){}
  static void deInitCupti() {}
  static void staticInit() {}
  static void setCounterAvailabilityImage(std::vector<uint8_t> /*img*/) {}
};
#endif // HAS_CUPTI_PROFILER

} // namespace KINETO_NAMESPACE
