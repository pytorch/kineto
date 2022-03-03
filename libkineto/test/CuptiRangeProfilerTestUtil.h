// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdlib.h>
#include <gtest/gtest.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiRangeProfilerApi.h"

namespace KINETO_NAMESPACE {

#if HAS_CUPTI_PROFILER

class MockCuptiRBProfilerSession : public CuptiRBProfilerSession {
 public:
  MockCuptiRBProfilerSession(int deviceId, CUcontext ctx)
    : CuptiRBProfilerSession(deviceId, ctx) {}

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
    runChecks();
  }

  void enable() override {
    enabled = true;
  }
  void disable() override {}

  CuptiProfilerResult evaluateMetrics(bool /*verbose*/) override {
    return result;
  }

protected:
  void startInternal(
      CUpti_ProfilerRange profilerRange,
      CUpti_ProfilerReplayMode profilerReplayMode) override {
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

  CuptiProfilerResult result;

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

#endif // HAS_CUPTI_PROFILER

} // namespace KINETO_NAMESPACE
