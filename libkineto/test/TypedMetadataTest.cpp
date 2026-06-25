/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/TypedMetadata.h"
#include "include/TypedMetadataJson.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

using namespace libkineto;

namespace {

constexpr MetadataField<int64_t> kCount{"count"};
constexpr MetadataField<std::vector<int64_t>> kIds{"ids"};
constexpr MetadataField<std::string> kLabel{"label"};
constexpr MetadataField<double> kRatio{"ratio"};
constexpr MetadataField<bool> kEnabled{"enabled"};
constexpr MetadataField<std::vector<std::string>> kNames{"names"};
constexpr MetadataField<std::pair<int64_t, int64_t>> kPair{"pair"};
constexpr MetadataField<std::string> kSpecialChars{"special"};

using RecordedValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<std::string>>;

class RecordingTypedMetadataVisitor : public ITypedMetadataVisitor {
 public:
  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    record(field, value);
  }

  void visitValue(const MetadataField<double>& field, double value) override {
    record(field, value);
  }

  void visitValue(const MetadataField<bool>& field, bool value) override {
    record(field, value);
  }

  void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) override {
    record(field, std::string{value});
  }

  void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& value) override {
    record(field, value);
  }

  void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& value) override {
    record(field, value);
  }

  void visitUnsupported(std::string_view /*name*/) override {}

  std::map<std::string, RecordedValue> values;

 private:
  template <typename T>
  void record(const MetadataField<T>& field, RecordedValue value) {
    values[std::string{field.name}] = std::move(value);
  }
};

class UnsupportedRecordingTypedMetadataVisitor final
    : public RecordingTypedMetadataVisitor {
 public:
  void visitUnsupported(std::string_view name) override {
    unsupportedFields.emplace_back(name);
  }

  std::vector<std::string> unsupportedFields;
};

} // namespace

TEST(TypedMetadataVisitorTest, VisitsTypedFields) {
  RecordingTypedMetadataVisitor recorder;
  ITypedMetadataVisitor& visitor = recorder;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kIds, std::vector<int64_t>{1, 2});
  visitor.visit(kLabel, std::string{"label"});
  visitor.visit(kRatio, 1.5);
  visitor.visit(kEnabled, true);
  visitor.visit(kNames, std::vector<std::string>{"a", "b"});

  EXPECT_EQ(std::get<int64_t>(recorder.values.at("count")), int64_t{5});
  EXPECT_EQ(
      std::get<std::vector<int64_t>>(recorder.values.at("ids")),
      std::vector<int64_t>({1, 2}));
  EXPECT_EQ(
      std::get<std::string>(recorder.values.at("label")), std::string{"label"});
  EXPECT_EQ(std::get<double>(recorder.values.at("ratio")), 1.5);
  EXPECT_EQ(std::get<bool>(recorder.values.at("enabled")), true);
  EXPECT_EQ(
      std::get<std::vector<std::string>>(recorder.values.at("names")),
      std::vector<std::string>({"a", "b"}));
}

TEST(TypedMetadataVisitorTest, SerializesVisitedFieldsToJson) {
  internal::JsonTypedMetadataVisitor jsonVisitor;
  ITypedMetadataVisitor& visitor = jsonVisitor;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kIds, std::vector<int64_t>{1, 2});
  visitor.visit(kLabel, std::string{"label"});
  visitor.visit(kRatio, 1.5);
  visitor.visit(kEnabled, true);
  visitor.visit(kNames, std::vector<std::string>{"a", "b"});
  visitor.visit(kSpecialChars, std::string{"quote \" slash \\ newline\n"});

  const auto json = std::move(jsonVisitor).json();

  EXPECT_NE(json.find("\"count\": 5"), std::string::npos);
  EXPECT_NE(json.find("\"ids\": [1, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"label\": \"label\""), std::string::npos);
  EXPECT_NE(json.find("\"ratio\": 1.5"), std::string::npos);
  EXPECT_NE(json.find("\"enabled\": true"), std::string::npos);
  EXPECT_NE(json.find("\"names\": [\"a\", \"b\"]"), std::string::npos);
  // String values pass through verbatim; output_json.cpp sanitizes the full
  // metadata fragment before appending it to the Chrome trace.
  EXPECT_NE(
      json.find("\"special\": \"quote \" slash \\ newline\n\""),
      std::string::npos);
}

TEST(TypedMetadataVisitorTest, FallsBackToVisitUnsupported) {
  UnsupportedRecordingTypedMetadataVisitor recorder;
  ITypedMetadataVisitor& visitor = recorder;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kPair, std::pair<int64_t, int64_t>{1, 2});

  EXPECT_EQ(std::get<int64_t>(recorder.values.at("count")), int64_t{5});
  EXPECT_EQ(recorder.unsupportedFields, std::vector<std::string>({"pair"}));
}
