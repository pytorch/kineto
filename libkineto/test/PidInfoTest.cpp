// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "include/ThreadUtil.h"

#include <atomic>
#include <thread>

#include <fmt/format.h>
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
