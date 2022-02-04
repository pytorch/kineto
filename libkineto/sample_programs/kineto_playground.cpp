// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <common/logging/logging.h>
#include <libkineto.h>

#include "kineto/libkineto/sample_programs/kineto_playground.cuh"

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
  LOG(INFO) << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.";
  trace->save(kFileName);
  return 0;
}

