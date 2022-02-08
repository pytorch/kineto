// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "src/Logger.h"
#include "src/CuptiCallbackApi.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>

using namespace std::chrono;
using namespace KINETO_NAMESPACE;
using namespace libkineto;

const size_t some_data = 42;

std::atomic<int> simple_cb_calls = 0;

void simple_cb(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {

  // simple arg check
  EXPECT_EQ(domain, CUPTI_CB_DOMAIN_RUNTIME_API);
  EXPECT_EQ(cbid, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  EXPECT_EQ(*reinterpret_cast<const size_t*>(cbInfo), some_data);

  simple_cb_calls++;
}

void atomic_cb(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* /*cbInfo)*/) {
  // do some atomics in a loop
  for (int i = 0; i < 1000; i++) {
    // would have used release consistency but this is fine
    simple_cb_calls++;
  }
}

void empty_cb(
    CUpti_CallbackDomain /*domain*/,
    CUpti_CallbackId /*cbid*/,
    const CUpti_CallbackData* /*cbInfo*/) {
}

TEST(CuptiCallbackApiTest, SimpleTest) {
  auto& api = CuptiCallbackApi::singleton();

  auto addSimpleCallback = [&]() -> bool {
    bool ret = api.registerCallback(
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CuptiCallbackApi::CUDA_LAUNCH_KERNEL,
        &simple_cb
    );
    return ret;
  };
  EXPECT_TRUE(addSimpleCallback()) << "Failed to add callback";

  // duplicate add should be okay
  EXPECT_TRUE(addSimpleCallback()) << "Failed to re-add callback";

  simple_cb_calls = 0;

  // simulate callback
  api.__callback_switchboard(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
      reinterpret_cast<const CUpti_CallbackData*>(&some_data));

  EXPECT_EQ(simple_cb_calls, 1);

  bool ret = api.deleteCallback(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CuptiCallbackApi::CUDA_LAUNCH_KERNEL,
      &simple_cb
  );

  EXPECT_TRUE(ret) << "Failed to remove callback";

  ret = api.deleteCallback(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CuptiCallbackApi::CUDA_LAUNCH_KERNEL,
      &atomic_cb
  );

  EXPECT_FALSE(ret) << "oops! deleted a callback that was never added";
}

TEST(CuptiCallbackApiTest, AllCallbacks) {
  auto& api = CuptiCallbackApi::singleton();

  auto testCallback = [&](
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid,
      CuptiCallbackApi::CuptiCallBackID kineto_cbid) -> bool {

    bool ret = api.registerCallback(domain, kineto_cbid, atomic_cb);
    EXPECT_TRUE(ret) << "Failed to add callback";

    if (!ret) {
      return false;
    }

    simple_cb_calls = 0;
    api.__callback_switchboard(domain, cbid, nullptr);
    EXPECT_EQ(simple_cb_calls, 1000);
    ret = simple_cb_calls == 1000;

    EXPECT_TRUE(api.deleteCallback(domain, kineto_cbid, atomic_cb));

    return ret;
  };

  EXPECT_TRUE(
      testCallback(
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
        CuptiCallbackApi::RESOURCE_CONTEXT_CREATED))
    << "Failed to run callback for RESOURCE_CONTEXT_CREATED";

  EXPECT_TRUE(
      testCallback(
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
        CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED))
    << "Failed to run callback for RESOURCE_CONTEXT_DESTROYED";

  EXPECT_TRUE(
      testCallback(
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
        CuptiCallbackApi::CUDA_LAUNCH_KERNEL))
    << "Failed to run callback for CUDA_LAUNCH_KERNEL";

}

TEST(CuptiCallbackApiTest, ContentionTest) {
  auto& api = CuptiCallbackApi::singleton();
  const CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RUNTIME_API;
  const CUpti_CallbackId cbid = CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
  const CuptiCallbackApi::CuptiCallBackID kineto_cbid =
    CuptiCallbackApi::CUDA_LAUNCH_KERNEL;

  bool ret = api.registerCallback(domain, kineto_cbid, empty_cb);
  EXPECT_TRUE(ret) << "Failed to add callback";

  const int iters = 10000;
  const int num_readers = 8;

  simple_cb_calls = 0;

  // simulate callbacks being executed on multiple threads in parallel
  //  during this interval add a new atomic_callback.
  //  this test ensured mutual exclusion is working fine
  auto read_fn = [&](int tid){
    auto start_ts = high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
      api.__callback_switchboard(domain, cbid, nullptr);
    }
    auto runtime_ms = duration_cast<milliseconds>(
      high_resolution_clock::now() - start_ts);
    LOG(INFO) << "th " << tid << " done in " << runtime_ms.count() << " ms";
  };


  std::vector<std::thread> read_ths;
  for (int i = 0; i< num_readers; i++) {
    read_ths.emplace_back(read_fn, i);
  }

  ret = api.registerCallback(domain, kineto_cbid, atomic_cb);
  EXPECT_TRUE(ret) << "Failed to add callback";

  for (auto& t : read_ths) {
    t.join();
  }

  //EXPECT_GT(simple_cb_calls, 0)
  //  << "Atomic callback should have been called at least once.";

  api.deleteCallback(domain, kineto_cbid, empty_cb);
  api.deleteCallback(domain, kineto_cbid, atomic_cb);
}

TEST(CuptiCallbackApiTest, Bechmark) {

  constexpr int iters = 1000;
  // atomic bench a number of times to get a baseline

  const CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RUNTIME_API;
  const CUpti_CallbackId cbid = CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
  const CuptiCallbackApi::CuptiCallBackID kineto_cbid =
    CuptiCallbackApi::CUDA_LAUNCH_KERNEL;

  LOG(INFO) << "Iteration count = " << iters;

  const bool use_empty = true;
  auto cbfn = use_empty ? &empty_cb : &atomic_cb;

  // warmup
  for (int i = 0; i < 50; i++) {
    (*cbfn)(domain, cbid, nullptr);
  }

  auto start_ts = high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    (*cbfn)(domain, cbid, nullptr);
  }
  auto delta_baseline_ns = duration_cast<nanoseconds>(
      high_resolution_clock::now() - start_ts);
  LOG(INFO) << "Baseline runtime  = " << delta_baseline_ns.count() << " ns";


  auto& api = CuptiCallbackApi::singleton();
  bool ret = api.registerCallback(domain, kineto_cbid, cbfn);
  EXPECT_TRUE(ret) << "Failed to add callback";

  // warmup
  for (int i = 0; i < 50; i++) {
    api.__callback_switchboard(domain, cbid, nullptr);
  }

  start_ts = high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    api.__callback_switchboard(domain, cbid, nullptr);
  }

  auto delta_callback_ns = duration_cast<nanoseconds>(
      high_resolution_clock::now() - start_ts);
  LOG(INFO) << "Callback runtime  = " << delta_callback_ns.count() << " ns";

  LOG(INFO) << "Callback runtime per iteration = " <<
    (delta_callback_ns.count() - delta_baseline_ns.count()) / (double) iters
    << " ns";

}
