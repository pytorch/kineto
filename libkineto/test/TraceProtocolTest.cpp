/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <string>

#include "TraceProtocol.h"

using namespace KINETO_NAMESPACE;

// ============================================================================
// extractJsonInt Tests
// ============================================================================

TEST(TraceProtocolTest, ExtractJsonIntBasic) {
  std::string const json = R"({"duration_ms": 500})";

  int const val = extractJsonInt(json, "duration_ms", 0);
  EXPECT_EQ(val, 500);
}

TEST(TraceProtocolTest, ExtractJsonIntMultipleKeys) {
  std::string const json =
      R"({"duration_ms": 500, "warmup_ms": 100, "count": 42})";

  int const duration = extractJsonInt(json, "duration_ms", 0);
  EXPECT_EQ(duration, 500);

  int const warmup = extractJsonInt(json, "warmup_ms", 0);
  EXPECT_EQ(warmup, 100);

  int const count = extractJsonInt(json, "count", 0);
  EXPECT_EQ(count, 42);
}

TEST(TraceProtocolTest, ExtractJsonIntMissingKey) {
  std::string const json = R"({"duration_ms": 500})";

  int const val = extractJsonInt(json, "missing_key", -1);
  EXPECT_EQ(val, -1);
}

TEST(TraceProtocolTest, ExtractJsonIntNegativeValue) {
  std::string const json = R"({"offset": -100})";

  int const val = extractJsonInt(json, "offset", 0);
  EXPECT_EQ(val, -100);
}

TEST(TraceProtocolTest, ExtractJsonIntZero) {
  std::string const json = R"({"value": 0})";

  int const val = extractJsonInt(json, "value", 999);
  EXPECT_EQ(val, 0);
}

TEST(TraceProtocolTest, ExtractJsonIntWhitespace) {
  std::string const json = R"({"duration_ms"  :  1000})";

  int const val = extractJsonInt(json, "duration_ms", 0);
  EXPECT_EQ(val, 1000);
}

// ============================================================================
// extractJsonBool Tests
// ============================================================================

TEST(TraceProtocolTest, ExtractJsonBoolTrue) {
  std::string const json = R"({"enabled": true})";

  bool const val = extractJsonBool(json, "enabled", false);
  EXPECT_TRUE(val);
}

TEST(TraceProtocolTest, ExtractJsonBoolFalse) {
  std::string const json = R"({"disabled": false})";

  bool const val = extractJsonBool(json, "disabled", true);
  EXPECT_FALSE(val);
}

TEST(TraceProtocolTest, ExtractJsonBoolMultipleBools) {
  std::string const json =
      R"({"record_shapes": true, "profile_memory": false, "with_stack": true})";

  bool const recordShapes = extractJsonBool(json, "record_shapes", false);
  EXPECT_TRUE(recordShapes);

  bool const profileMemory = extractJsonBool(json, "profile_memory", true);
  EXPECT_FALSE(profileMemory);

  bool const withStack = extractJsonBool(json, "with_stack", false);
  EXPECT_TRUE(withStack);
}

TEST(TraceProtocolTest, ExtractJsonBoolMissingKey) {
  std::string const json = R"({"enabled": true})";

  bool val = extractJsonBool(json, "missing_key", true);
  EXPECT_TRUE(val); // Returns default

  val = extractJsonBool(json, "missing_key", false);
  EXPECT_FALSE(val); // Returns default
}

TEST(TraceProtocolTest, ExtractJsonBoolWhitespace) {
  std::string const json = R"({"enabled"  :  true})";

  bool const val = extractJsonBool(json, "enabled", false);
  EXPECT_TRUE(val);
}

// ============================================================================
// extractJsonString Tests
// ============================================================================

TEST(TraceProtocolTest, ExtractJsonStringBasic) {
  std::string const json = R"({"trace_id": "abc-123"})";

  std::string const val = extractJsonString(json, "trace_id", "");
  EXPECT_EQ(val, "abc-123");
}

TEST(TraceProtocolTest, ExtractJsonStringPath) {
  std::string const json = R"({"output_dir": "/tmp/kineto_traces"})";

  std::string const val = extractJsonString(json, "output_dir", "");
  EXPECT_EQ(val, "/tmp/kineto_traces");
}

TEST(TraceProtocolTest, ExtractJsonStringEmptyString) {
  std::string const json = R"({"trace_id": ""})";

  std::string const val = extractJsonString(json, "trace_id", "default");
  EXPECT_EQ(val, "");
}

TEST(TraceProtocolTest, ExtractJsonStringMissingKey) {
  std::string const json = R"({"trace_id": "abc"})";

  std::string const val = extractJsonString(json, "missing_key", "default");
  EXPECT_EQ(val, "default");
}

TEST(TraceProtocolTest, ExtractJsonStringWhitespace) {
  std::string const json = R"({"trace_id"  :  "xyz-789"})";

  std::string const val = extractJsonString(json, "trace_id", "");
  EXPECT_EQ(val, "xyz-789");
}

TEST(TraceProtocolTest, ExtractJsonStringUUID) {
  std::string const json =
      R"({"trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"})";

  std::string const val = extractJsonString(json, "trace_id", "");
  EXPECT_EQ(val, "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
}

// ============================================================================
// buildConfigString Tests
// ============================================================================

TEST(TraceProtocolTest, BuildConfigStringFullConfig) {
  std::string const config = buildConfigString(
      1000, // durationMs
      "CUDA,CPU", // activities
      true, // recordShapes
      false, // profileMemory
      true, // withStack
      false, // withFlops
      true // withModules
  );

  // Verify key config values are present
  EXPECT_NE(config.find("ACTIVITIES_DURATION_MSECS=1000"), std::string::npos);
  EXPECT_NE(config.find("ACTIVITIES=CUDA,CPU"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_REPORT_INPUT_SHAPES=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_WITH_STACK=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_WITH_MODULES=true"), std::string::npos);

  // These should NOT be present since they are false
  EXPECT_EQ(config.find("PROFILE_PROFILE_MEMORY=true"), std::string::npos);
  EXPECT_EQ(config.find("PROFILE_WITH_FLOPS=true"), std::string::npos);
}

TEST(TraceProtocolTest, BuildConfigStringMinimalConfig) {
  std::string const config = buildConfigString(
      500, // durationMs
      "", // activities (empty)
      false, // recordShapes
      false, // profileMemory
      false, // withStack
      false, // withFlops
      false // withModules
  );

  // Duration should be present
  EXPECT_NE(config.find("ACTIVITIES_DURATION_MSECS=500"), std::string::npos);

  // Empty activities should not add ACTIVITIES line
  EXPECT_EQ(config.find("ACTIVITIES="), std::string::npos);

  // No boolean flags should be set
  EXPECT_EQ(config.find("PROFILE_REPORT_INPUT_SHAPES"), std::string::npos);
}

TEST(TraceProtocolTest, BuildConfigStringAllBooleansTrue) {
  std::string const config = buildConfigString(
      100, // durationMs
      "CUDA", // activities
      true, // recordShapes
      true, // profileMemory
      true, // withStack
      true, // withFlops
      true // withModules
  );

  EXPECT_NE(config.find("PROFILE_REPORT_INPUT_SHAPES=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_PROFILE_MEMORY=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_WITH_STACK=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_WITH_FLOPS=true"), std::string::npos);
  EXPECT_NE(config.find("PROFILE_WITH_MODULES=true"), std::string::npos);
}
