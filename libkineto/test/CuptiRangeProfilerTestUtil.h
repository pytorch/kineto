/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <unordered_map>
#include <gtest/gtest.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiRangeProfilerApi.h"

namespace KINETO_NAMESPACE {

#if HAS_CUPTI_RANGE_PROFILER

class MockCuptiRBProfilerSession : public CuptiRBProfilerSession {
 public:
  explicit MockCuptiRBProfilerSession(const CuptiRangeProfilerOptions& opts)
    : CuptiRBProfilerSession(opts) {}

  void beginPass() override {
    LOG(INFO) << " Mock CUPTI begin pass";
    passes_started++;
  }

  bool endPass() override {
    passes_ended++;
    return true;
  }

  void flushCounterData() override {}

  void pushRange(const std::string& rangeName) override {
    LOG(INFO) << " Mock CUPTI pushrange ( " << rangeName << " )";
    ranges_started++;
  }

  void popRange() override {
    LOG(INFO) << " Mock CUPTI poprange";
    ranges_ended++;
  }

  void stop() override {
    profilerStopTs_ = std::chrono::high_resolution_clock::now();
    runChecks();
  }

  void enable() override {
    enabled = true;
  }
  void disable() override {}

  CuptiProfilerResult evaluateMetrics(bool /*verbose*/) override {
    return getResults()[deviceId()];
  }

protected:
  void startInternal(
      CUpti_ProfilerRange profilerRange,
      CUpti_ProfilerReplayMode profilerReplayMode) override {
    profilerStartTs_ = std::chrono::high_resolution_clock::now();
    curRange_ = profilerRange;
    curReplay_ = profilerReplayMode;
  }

private:
  void runChecks() {
    EXPECT_EQ(passes_started, passes_ended);
    EXPECT_EQ(ranges_started, ranges_ended);
  }

 public:
  int passes_started = 0;
  int passes_ended = 0;
  int ranges_started = 0;
  int ranges_ended = 0;
  bool enabled = false;

  static std::unordered_map<int, CuptiProfilerResult>& getResults();
};

struct MockCuptiRBProfilerSessionFactory : ICuptiRBProfilerSessionFactory {
  std::unique_ptr<CuptiRBProfilerSession> make(
      const CuptiRangeProfilerOptions& _opts) override {
    auto opts = _opts;
    opts.unitTest = true;
    return std::make_unique<MockCuptiRBProfilerSession>(opts);
  }

  MockCuptiRBProfilerSession* asDerived(CuptiRBProfilerSession* base) {
    return dynamic_cast<MockCuptiRBProfilerSession*>(base);
  }
};

inline void simulateCudaContextCreate(CUcontext context, uint32_t dev) {
  testing::trackCudaCtx(
      context, dev, CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
}

inline void simulateCudaContextDestroy(CUcontext context, uint32_t dev) {
  testing::trackCudaCtx(
      context, dev, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);
}

inline void simulateKernelLaunch(
    CUcontext context, const std::string& kernelName) {
  testing::trackCudaKernelLaunch(context, kernelName.c_str());
}

#endif // HAS_CUPTI_RANGE_PROFILER

} // namespace KINETO_NAMESPACE
