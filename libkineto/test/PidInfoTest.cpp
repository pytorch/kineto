/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/ThreadName.h"

#include <atomic>
#include <sys/syscall.h>
#include <thread>

#include <fmt/format.h>
#include <gtest/gtest.h>

using namespace KINETO_NAMESPACE;

TEST(ThreadNameTest, setAndGet) {
  setThreadName("ThreadNameTest");
  EXPECT_EQ(getThreadName(getpid()), "ThreadNameTest");

  setThreadName("");
  EXPECT_EQ(getThreadName(getpid()), "");

  // Spaces etc are ok
  setThreadName("Name w/ spaces");
  EXPECT_EQ(getThreadName(getpid()), "Name w/ spaces");

  // More than 16 chars is not OK
  setThreadName("More than 16 characters");
  EXPECT_EQ(getThreadName(getpid()), "Name w/ spaces");
}

TEST(ThreadNameTest, invalidThread) {
  EXPECT_EQ(getThreadName(123456789), "Unknown");
}

TEST(ThreadNameTest, otherThread) {
  std::atomic_bool stop_flag;
  std::atomic_int tid = 0;
  std::thread thread([&stop_flag, &tid]() {
      setThreadName("New Thread");
      tid = syscall(SYS_gettid);
      while (!stop_flag) {
        sleep(1);
      }
  });
  while (!tid) {
    sleep(1);
  }
  EXPECT_EQ(getThreadName(tid), "New Thread");
  stop_flag = true;
  thread.join();
}

TEST(ThreadNameTest, deadThread) {
  std::atomic_bool stop_flag;
  std::atomic_int tid = 0;
  std::thread thread([&stop_flag, &tid]() {
      setThreadName("New Thread");
      tid = syscall(SYS_gettid);
      while (!stop_flag) {
        sleep(1);
      }
  });
  while (!tid) {
    sleep(1);
  }
  stop_flag = true;
  thread.join();
  EXPECT_EQ(getThreadName(tid), "Unknown");
}

