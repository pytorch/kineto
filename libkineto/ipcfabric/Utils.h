/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace dynolog {
namespace ipcfabric {

// struct to register libkineto process
struct LibkinetoContext {
  // gpu id of the process running on
  int32_t gpu;
  // pid to register to dynolog
  pid_t pid;
  // job id of the process
  int64_t jobid;
};

// struct to request libkineto ondemand config
struct LibkinetoRequest {
  // type of libkineto config
  int type;
  // size of pids
  int n;
  // job id of the libkineto process
  int64_t jobid;
  // pids of the process and its ancestors
  int32_t pids[];
};

} // namespace ipcfabric
} // namespace dynolog
