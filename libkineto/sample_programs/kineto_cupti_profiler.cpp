/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <thread>
#include <iostream>

#include <libkineto.h>

// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "kineto_playground.cuh"

using namespace kineto;

static const std::string kFileName = "/tmp/kineto_playground_trace.json";

int main() {
  warmup();

  // Kineto config
  std::set<libkineto::ActivityType> types_cupti_prof = {
    libkineto::ActivityType::CUDA_PROFILER_RANGE,
  };

  libkineto_init(false, true);
  libkineto::api().initProfilerIfRegistered();

  // Use a special kineto__cuda_core_flop metric that counts individual
  // CUDA core floating point instructions by operation type (fma,fadd,fmul,dadd ...)
  // You can also use kineto__tensor_core_insts or any metric
  // or any metric defined by CUPTI Profiler below
  //   https://docs.nvidia.com/cupti/Cupti/r_main.html#r_profiler

  std::string profiler_config = "ACTIVITIES_WARMUP_PERIOD_SECS=0\n "
    "CUPTI_PROFILER_METRICS=kineto__cuda_core_flops\n "
    "CUPTI_PROFILER_ENABLE_PER_KERNEL=true";

  auto& profiler = libkineto::api().activityProfiler();
  profiler.prepareTrace(types_cupti_prof, profiler_config);

  // Good to warm up after prepareTrace to get cupti initialization to settle
  warmup();

  profiler.startTrace();
  basicMemcpyToDevice();
  compute();
  basicMemcpyFromDevice();

  auto trace = profiler.stopTrace();
  std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.";
  trace->save(kFileName);
  return 0;
}
