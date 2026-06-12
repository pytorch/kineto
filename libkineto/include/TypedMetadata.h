/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace libkineto {

using TypedValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<std::string>>;

template <typename T>
struct MetadataField {
  using FieldType = T;

  std::string_view name;
};

/*
 * TypedMetadata is a per-activity map for structured metadata with
 * field-level type checking.
 *
 * Usage:
 *   // Declare each field once. FieldType is int64_t here, and set/get enforce
 *   // that declared type.
 *   inline constexpr MetadataField<int64_t> kCorrelation{"correlation"};
 *
 *   // Producer: populate metadata from ITraceActivity::typedMetadata().
 *   TypedMetadata metadata;
 *   metadata.set(kCorrelation, int64_t{activity.correlationId});
 *
 *   // Consumer: read a known field with its declared type.
 *   std::optional<int64_t> correlation = metadata.get(kCorrelation);
 *
 *   // Consumer: iterate over all fields for generic output conversion.
 *   metadata.visit([](std::string_view name, const TypedValue& value) {
 *     ...
 *   });
 */
class TypedMetadata {
 public:
  template <typename T, typename V>
  void set(const MetadataField<T>& field, V&& value) {
    using FieldType = typename MetadataField<T>::FieldType;
    static_assert(std::is_same_v<FieldType, std::decay_t<V>>, "value type must match field's declared type");
    entries_.insert_or_assign(std::string{field.name}, TypedValue{std::forward<V>(value)});
  }

  [[nodiscard]] std::optional<TypedValue> get(std::string_view key) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  template <typename T>
  [[nodiscard]] std::optional<typename MetadataField<T>::FieldType> get(const MetadataField<T>& field) const {
    using FieldType = typename MetadataField<T>::FieldType;
    auto it = entries_.find(field.name);
    if (it == entries_.end()) {
      return std::nullopt;
    }
    return std::get<FieldType>(it->second);
  }

  template <typename Fn>
  void visit(Fn&& fn) const {
    for (const auto& [name, value] : entries_) {
      fn(std::string_view{name}, value);
    }
  }

  [[nodiscard]] bool empty() const {
    return entries_.empty();
  }

 private:
  std::map<std::string, TypedValue, std::less<>> entries_;
};

} // namespace libkineto
