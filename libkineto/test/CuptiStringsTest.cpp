/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/cupti_strings.h"

using namespace KINETO_NAMESPACE;

TEST(CuptiStringsTest, Valid) {
  ASSERT_STREQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_INVALID), "INVALID");
  ASSERT_STREQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020),
      "cudaDriverGetVersion");
  ASSERT_STREQ(runtimeCbidName
      (CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020),
      "cudaDeviceSynchronize");
  ASSERT_STREQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_ptsz_v11000),
      "cudaStreamSetAttribute_ptsz");
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
  ASSERT_STREQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060),
      "cudaLaunchKernelExC");
#endif
}

TEST(CuptiStringsTest, Invalid) {
  ASSERT_STREQ(runtimeCbidName(-1), "INVALID");
  // We can't actually use CUPTI_RUNTIME_TRACE_CBID_SIZE here until we
  // auto-generate the string table, since it may have more entries than
  // the enum in the version used to compile.
  ASSERT_STREQ(runtimeCbidName(1000), "INVALID");
}
