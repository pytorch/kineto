// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <signal.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace libkineto {

class external_api {
 public:
  struct OpDetails {
    int64_t startTime;
    int64_t endTime;
    uint64_t correlationId;
    std::string opType;
    int deviceId;
    pthread_t threadId;
    std::string inputDims;
    std::string inputTypes;
    std::string arguments;
    std::string outputDims;
    std::string outputTypes;
    std::string inputNames;
    std::string outputNames;
  };
  struct CpuTraceBuffer {
    std::string netName;
    int64_t startTime;
    int64_t endTime;
    int gpuOpCount;
    std::vector<OpDetails> ops;
  };

  static void transferCpuTrace(std::unique_ptr<CpuTraceBuffer> cpuTrace);
  static void pushCorrelationID(int id);
  static void popCorrelationID();

  static inline int64_t timeSinceEpoch(
      const std::chrono::time_point<std::chrono::high_resolution_clock>& t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               t.time_since_epoch())
        .count();
  }

  // initialize the start/stop profilers
  static void initialize(
      std::function<void(std::unique_ptr<CpuTraceBuffer>)> transferFunc,
      std::function<void(int)> pushCorrelationIDFunc,
      std::function<void(void)> popCorrelationIDFunc,
      std::function<bool(const std::string&)> netNameFilterFunc);

  // Return whether the start/stopProfiler have been registered
  static inline bool isInitialized() {
    // Lazy init
    initLibkineto();
    return !!getInstance().transferCpuTrace_;
  }

  static inline bool profileRequestActive() {
    // Lazy init
    initLibkineto();
    return getInstance().requestActive_;
  }

  // Has libkineto been loaded, i.e., does server
  // have libcupti installed and libkineto active?
  static bool isLoaded();

  // Tell libkineto the external API is supported.
  // If this is not called, libkineto will fall back to GPU-only tracing.
  static void registerTracer(std::function<void(void)> init_func);

  // Profile net with this name
  static bool enableForNet(const std::string& name);

  // Profile nets with at least this many ops
  static int netSizeThreshold();
  static void setNetSizeThreshold(int gpu_ops);

  // Used by libkineto
  static void setProfileRequestActive(bool active);
  static void requestOnDemandProfile();
  static void setLoaded(std::function<void(void)> init_func);
  static bool isSupported();

 private:
  // Allow initialization to be delayed until actually needed
  static inline void initLibkineto() {
    if (getInstance().libkinetoInit_) {
      getInstance().libkinetoInit_();
      getInstance().libkinetoInit_ = nullptr;
    }
  }

  // Called when external tracer is registered and libkineto has been loaded
  static void initTracer();

  // singeleton design, only one external_api instance in the program
  static external_api& getInstance();
  // callback functions for the profiler to register
  std::function<void(std::unique_ptr<CpuTraceBuffer>)> transferCpuTrace_;
  std::function<void(int)> pushCorrelationID_;
  std::function<void(void)> popCorrelationID_;
  std::function<bool(const std::string&)> netNameFilter_;
  std::function<void(void)> externalInit_;
  std::function<void(void)> libkinetoInit_;
  pthread_t externalInitThread_{0};
  bool isLoaded_ = false;
  std::atomic_bool requestActive_{false};
  std::atomic_int netSizeThreshold_{0};
  std::mutex mutex_;
};

} // namespace libkineto
