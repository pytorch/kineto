/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/ThreadUtil.h"

#include <atomic>
#include <thread>

#include <fmt/format.h>
#include <gtest/gtest.h>

using namespace KINETO_NAMESPACE;

TEST(ThreadNameTest, setAndGet) {
  setThreadName("ThreadNameTest");
  EXPECT_EQ(getThreadName(systemThreadId()), "ThreadNameTest");

  setThreadName("");
  EXPECT_EQ(getThreadName(systemThreadId()), "");

  // Spaces etc are ok
  setThreadName("Name w/ spaces");
  EXPECT_EQ(getThreadName(systemThreadId()), "Name w/ spaces");

  // More than 16 chars is not OK
  setThreadName("More than 16 characters");
  EXPECT_EQ(getThreadName(systemThreadId()), "Name w/ spaces");
}

TEST(ThreadNameTest, invalidThread) {
  EXPECT_EQ(getThreadName(123456789), "Unknown");
}

TEST(ThreadNameTest, otherThread) {
  std::atomic_bool stop_flag;
  std::atomic_int tid = 0;
  std::thread thread([&stop_flag, &tid]() {
      setThreadName("New Thread");
      tid = systemThreadId();
      while (!stop_flag) {}
  });
  while (!tid) {}
  EXPECT_EQ(getThreadName(tid), "New Thread");
  stop_flag = true;
  thread.join();
}

TEST(ThreadNameTest, deadThread) {
  std::atomic_bool stop_flag;
  std::atomic_int tid = 0;
  std::thread thread([&stop_flag, &tid]() {
      setThreadName("New Thread");
      tid = systemThreadId();
      while (!stop_flag) {}
  });
  while (!tid) {}
  stop_flag = true;
  thread.join();
  // There appears to be a delay before the thread info is
  // removed from proc - we can therefore expect either
  // "Unknown" or "New Thread" to be returned.
  std::string name = getThreadName(tid);
  EXPECT_TRUE(name == "Unknown" || name == "New Thread")
    << "Where name = " << name;
}
