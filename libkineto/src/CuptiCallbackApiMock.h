/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Provides data structures to mock CUPTI Callback API
#ifndef HAS_CUPTI

enum CUpti_CallbackDomain {
  CUPTI_CB_DOMAIN_RESOURCE,
  CUPTI_CB_DOMAIN_RUNTIME_API,
};
enum CUpti_CallbackId {
  CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
  CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
  CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
};

using CUcontext = void*;

struct CUpti_ResourceData {
  CUcontext context;
};

constexpr int CUPTI_API_ENTER = 0;
constexpr int CUPTI_API_EXIT = 0;

struct CUpti_CallbackData {
  CUcontext context;
  const char* symbolName;
  int callbackSite;
};
#endif // HAS_CUPTI
