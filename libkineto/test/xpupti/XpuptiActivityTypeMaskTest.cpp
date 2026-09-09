/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/plugin/xpupti/XpuptiActivityTypeMask.h"

#include <gtest/gtest.h>

#include <set>
#include <stdexcept>

namespace KN = KINETO_NAMESPACE;
using libkineto::ActivityType;

TEST(ActivityTypeMaskTest, ContainsExactlyTheSelectedTypes) {
  const std::set<ActivityType> selected = {
      ActivityType::XPU_DRIVER,
      ActivityType::CONCURRENT_KERNEL,
      ActivityType::GPU_MEMCPY};
  const KN::ActivityTypeMask mask(selected);

  for (int type = 0; type < libkineto::activityTypeCount; ++type) {
    const auto activityType = static_cast<ActivityType>(type);
    EXPECT_EQ(mask.contains(activityType), selected.contains(activityType))
        << "type " << type;
  }
}

// The bitset is sized from the enum, so the last enumerator is the case that
// would fall off a mask sized by hand.
TEST(ActivityTypeMaskTest, HoldsTheHighestActivityType) {
  constexpr auto highest =
      static_cast<ActivityType>(libkineto::activityTypeCount - 1);
  const KN::ActivityTypeMask mask({highest});

  EXPECT_TRUE(mask.contains(highest));
  EXPECT_FALSE(mask.contains(static_cast<ActivityType>(0)));
}

TEST(ActivityTypeMaskTest, HoldsEveryActivityTypeAtOnce) {
  std::set<ActivityType> all;
  for (int type = 0; type < libkineto::activityTypeCount; ++type) {
    all.insert(static_cast<ActivityType>(type));
  }
  const KN::ActivityTypeMask mask(all);

  for (const auto activityType : all) {
    EXPECT_TRUE(mask.contains(activityType))
        << "type " << static_cast<int>(activityType);
  }
}

TEST(ActivityTypeMaskTest, EmptySelectionContainsNothing) {
  const KN::ActivityTypeMask mask(std::set<ActivityType>{});

  EXPECT_FALSE(mask.contains(ActivityType::XPU_RUNTIME));
  EXPECT_FALSE(mask.contains(ActivityType::XPU_DRIVER));
}

// A value the enum does not define is a caller bug, not a silent false: the
// bitset reports it instead of shifting past its own width.
TEST(ActivityTypeMaskTest, RejectsAValueOutsideTheEnum) {
  const KN::ActivityTypeMask mask({ActivityType::XPU_DRIVER});
  const auto beyondEnum =
      static_cast<ActivityType>(libkineto::activityTypeCount);

  EXPECT_THROW(mask.contains(beyondEnum), std::out_of_range);
  EXPECT_THROW(KN::ActivityTypeMask({beyondEnum}), std::out_of_range);
}
