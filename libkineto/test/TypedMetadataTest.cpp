/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/TypedMetadata.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace libkineto;

namespace {

constexpr MetadataField<int64_t> kCount{"count"};
constexpr MetadataField<int64_t> kMissing{"missing"};
constexpr MetadataField<std::vector<int64_t>> kIds{"ids"};
constexpr MetadataField<std::string> kLabel{"label"};
constexpr MetadataField<double> kRatio{"ratio"};
constexpr MetadataField<bool> kEnabled{"enabled"};
constexpr MetadataField<std::vector<std::string>> kNames{"names"};

} // namespace

TEST(TypedMetadataTest, StoresAndRetrievesTypedValues) {
  TypedMetadata metadata;
  metadata.set(kCount, int64_t{5});
  metadata.set(kIds, std::vector<int64_t>{1, 2});
  metadata.set(kLabel, std::string{"label"});
  metadata.set(kRatio, 1.5);
  metadata.set(kEnabled, true);
  metadata.set(kNames, std::vector<std::string>{"a", "b"});

  EXPECT_FALSE(metadata.empty());
  EXPECT_EQ(metadata.get(kCount), int64_t{5});
  EXPECT_EQ(metadata.get(kIds), std::vector<int64_t>({1, 2}));
  EXPECT_EQ(metadata.get(kLabel), std::string{"label"});
  EXPECT_EQ(metadata.get(kRatio), 1.5);
  EXPECT_EQ(metadata.get(kEnabled), true);
  EXPECT_EQ(metadata.get(kNames), std::vector<std::string>({"a", "b"}));
  EXPECT_FALSE(metadata.get(kMissing).has_value());

  auto raw = metadata.get("count");
  ASSERT_TRUE(raw.has_value());
  EXPECT_EQ(std::get<int64_t>(*raw), int64_t{5});
}

TEST(TypedMetadataTest, VisitsStoredValuesByName) {
  TypedMetadata metadata;
  metadata.set(kCount, int64_t{5});
  metadata.set(kLabel, std::string{"label"});

  std::map<std::string, TypedValue> values;
  metadata.visit([&](std::string_view name, const TypedValue& value) {
    values.emplace(std::string{name}, value);
  });

  EXPECT_EQ(std::get<int64_t>(values.at("count")), int64_t{5});
  EXPECT_EQ(std::get<std::string>(values.at("label")), std::string{"label"});
}

TEST(TypedMetadataTest, OverwritesDuplicateFields) {
  TypedMetadata metadata;
  metadata.set(kCount, int64_t{5});
  metadata.set(kCount, int64_t{6});

  EXPECT_EQ(metadata.get(kCount), int64_t{6});
}
