/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <kineto_playground.cuh>

#include <libkineto.h>

#include <string>

using namespace kineto;

static const std::string kFileName = "/tmp/kineto_playground_trace.json";

int main() {
  warmup();

  // Kineto config

  // Empty types set defaults to all types
  std::set<libkineto::ActivityType> types;

  auto& profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  profiler.prepareTrace(types);

  // Good to warm up after prepareTrace to get cupti initialization to settle
  warmup();
  profiler.startTrace();
  playground();

  auto trace = profiler.stopTrace();
  trace->save(kFileName);
  return 0;
}
