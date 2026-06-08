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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/Config.h"
#include "include/GenericTraceActivity.h"
#include "include/TraceSpan.h"
#include "src/output_json.h"

using namespace KINETO_NAMESPACE;
using namespace libkineto;

namespace {

// Emit a trace carrying a single WARNING log string and return the file text.
std::string traceWithWarning(const std::string& warning) {
  const std::string path =
      ::testing::TempDir() + "kineto_output_json_test.json";
  ChromeTraceLogger logger(path);
  logger.handleTraceStart({}, /*device_properties=*/"");
  std::unordered_map<std::string, std::vector<std::string>> metadata;
  metadata["WARNING"] = {warning};
  Config config;
  logger.finalizeTrace(config, /*buffers=*/nullptr, /*endTime=*/1000, metadata);

  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Exposes the protected finalizeTrace(endTime, metadata) overload so the test
// can flush a trace without constructing a Config / ActivityBuffers.
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

void writeTraceWithEventName(
    const std::string& filename,
    const std::string& eventName) {
  TraceSpan span(0, 0, "test_span");
  GenericTraceActivity act(span, ActivityType::CPU_OP, eventName);
  act.startTime = 100;
  act.endTime = 200;
  act.device = 0;
  act.resource = 0;

  TestableChromeTraceLogger logger(filename);
  logger.handleTraceStart({}, "");
  logger.handleGenericActivity(act);
  std::unordered_map<std::string, std::vector<std::string>> metadata;
  logger.finalizeTrace(/*endTime=*/300, metadata);
}

} // namespace

TEST(OutputJsonTest, NeutralizesQuotesInLogStrings) {
  const std::string out = traceWithWarning(R"(trace_id not hex: "abc")");
  EXPECT_NE(out.find("trace_id not hex: 'abc'"), std::string::npos);
}

TEST(OutputJsonTest, DropsControlCharsInLogStrings) {
  const std::string out = traceWithWarning("abc\tdef\rghi\n");
  EXPECT_NE(out.find("abcdefghi"), std::string::npos);
}

// Reproduces pytorch/pytorch#146900: with with_stack=True an event name can
// contain `"` (e.g. torch.compile/dynamo-generated co_filenames such as
// `{"device": "cpu"}`). The chrome-trace writer must JSON-escape those quotes
// so the emitted trace remains valid JSON and the name round-trips intact.
TEST(OutputJsonTest, EventNameWithQuotesProducesValidJson) {
  const std::string filename = "/tmp/kineto_output_json_quotes_test.json";
  const std::string eventName =
      R"({"device": "cpu", "dtype": "float32"}(12): run)";

  writeTraceWithEventName(filename, eventName);

  const std::string jsonStr = readFile(filename);

  // Must parse without throwing (previously failed with a JSON parse error).
  nlohmann::json data;
  ASSERT_NO_THROW(data = nlohmann::json::parse(jsonStr));

  // The name must round-trip back to the original, unescaped value.
  bool found = false;
  for (const auto& event : data["traceEvents"]) {
    if (event.contains("ph") && event["ph"] == "X" && event.contains("name") &&
        event["name"].get<std::string>() == eventName) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Event with unescaped name not found in trace";

  std::remove(filename.c_str());
}

// A plain event name (no quotes) must be unaffected by the escaping.
TEST(OutputJsonTest, PlainEventNameIsUnchanged) {
  const std::string filename = "/tmp/kineto_output_json_plain_test.json";
  const std::string eventName = "aten::addmm";

  writeTraceWithEventName(filename, eventName);

  const std::string jsonStr = readFile(filename);
  nlohmann::json data;
  ASSERT_NO_THROW(data = nlohmann::json::parse(jsonStr));

  bool found = false;
  for (const auto& event : data["traceEvents"]) {
    if (event.contains("ph") && event["ph"] == "X" && event.contains("name") &&
        event["name"].get<std::string>() == eventName) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);

  std::remove(filename.c_str());
}
