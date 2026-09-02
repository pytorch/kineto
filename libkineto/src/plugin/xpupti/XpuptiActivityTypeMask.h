/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"

#include <bitset>
#include <cstddef>
#include <set>

namespace KINETO_NAMESPACE {

// A set of activity types held as a bitset: built once, queried per record.
// Sized from the enum, so activity types can be added without outgrowing it,
// and a value outside the enum throws instead of shifting out of range.
class ActivityTypeMask {
 public:
  ActivityTypeMask() = default;

  explicit ActivityTypeMask(const std::set<ActivityType>& types) {
    for (const auto type : types) {
      bits_.set(index(type));
    }
  }

  bool contains(ActivityType type) const {
    return bits_.test(index(type));
  }

 private:
  static constexpr size_t index(ActivityType type) {
    return static_cast<size_t>(type);
  }

  std::bitset<libkineto::activityTypeCount> bits_;
};

} // namespace KINETO_NAMESPACE
