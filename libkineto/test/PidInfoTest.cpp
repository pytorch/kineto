/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/ThreadUtil.h"

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

using namespace KINETO_NAMESPACE;

TEST(ThreadNameTest, setAndGet) {
  setThreadName("ThreadNameTest");
  EXPECT_EQ(getThreadName(), "ThreadNameTest");

  setThreadName("");
  EXPECT_EQ(getThreadName(), "");

  // Spaces etc are ok
  setThreadName("Name w/ spaces");
  EXPECT_EQ(getThreadName(), "Name w/ spaces");

  // More than 16 chars is not OK
  setThreadName("More than 16 characters");
  EXPECT_EQ(getThreadName(), "Name w/ spaces");
}
