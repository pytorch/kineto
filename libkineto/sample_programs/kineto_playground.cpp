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
#include <iostream>

#include <libkineto.h>

// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "kineto_playground.cuh"

using namespace kineto;

static const std::string kFileName = "/tmp/kineto_playground_trace.json";

int main() {
  warmup();

  // Kineto config

  // Empty types set defaults to all types
  std::set<libkineto::ActivityType> types;
  libkineto_init(false, true);
  libkineto::api().initProfilerIfRegistered();

  auto& profiler = libkineto::api().activityProfiler();
  profiler.prepareTrace(types);

  // Good to warm up after prepareTrace to get cupti initialization to settle
  warmup();
  profiler.startTrace();
  playground();

  auto trace = profiler.stopTrace();
  std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.";
  trace->save(kFileName);
  return 0;
}
