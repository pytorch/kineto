/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "include/GenericTraceActivity.h"
#include "include/TraceSpan.h"
#include "src/output_json.h"

using namespace KINETO_NAMESPACE;
using namespace libkineto;

namespace {

// Exposes the protected finalizeTrace(endTime) overload so the test can flush a
// trace without constructing a Config / ActivityBuffers.
class TestableChromeTraceLogger : public ChromeTraceLogger {
 public:
  explicit TestableChromeTraceLogger(const std::string& file)
      : ChromeTraceLogger(file) {}
  using ChromeTraceLogger::finalizeTrace;
};

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  return std::string(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Write a trace containing a single CPU-op event with the given name, parse it
// back, and assert the trace is valid JSON and the name round-trips exactly.
void expectEventNameRoundTrips(const std::string& eventName) {
  const std::string filename = ::testing::TempDir() + "kineto_output_json.json";

  TraceSpan span(0, 0, "test_span");
  GenericTraceActivity act(span, ActivityType::CPU_OP, eventName);
  act.startTime = 100;
  act.endTime = 200;
  act.device = 0;
  act.resource = 0;

  TestableChromeTraceLogger logger(filename);
  logger.handleTraceStart({}, "");
  logger.handleGenericActivity(act);
  logger.finalizeTrace(/*endTime=*/300);

  nlohmann::json data;
  ASSERT_NO_THROW(data = nlohmann::json::parse(readFile(filename)));

  bool found = false;
  for (const auto& event : data["traceEvents"]) {
    if (event.contains("ph") && event["ph"] == "X" && event.contains("name") &&
        event["name"].get<std::string>() == eventName) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "event name did not round-trip: " << eventName;

  std::remove(filename.c_str());
}

} // namespace

// Reproduces pytorch/pytorch#146900: with with_stack=True an event name can
// contain `"` (e.g. dynamo co_filenames like {"device": "cpu"}). The writer
// must JSON-escape those quotes so the trace stays valid JSON.
TEST(OutputJsonTest, EventNameWithQuotesProducesValidJson) {
  expectEventNameRoundTrips(
      R"({"device": "cpu", "dtype": "float32"}(12): run)");
}

TEST(OutputJsonTest, PlainEventNameIsUnchanged) {
  expectEventNameRoundTrips("aten::addmm");
}
