/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/Config.h"
#include "src/output_json.h"

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace KINETO_NAMESPACE;

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
} // namespace

TEST(OutputJsonTest, NeutralizesQuotesInLogStrings) {
  const std::string out = traceWithWarning(R"(trace_id not hex: "abc")");
  EXPECT_NE(out.find("trace_id not hex: 'abc'"), std::string::npos);
}

TEST(OutputJsonTest, DropsControlCharsInLogStrings) {
  const std::string out = traceWithWarning("abc\tdef\rghi\n");
  EXPECT_NE(out.find("abcdefghi"), std::string::npos);
}
