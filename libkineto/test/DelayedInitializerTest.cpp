/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace {

class TestDelayedInitializer {
 public:
  explicit TestDelayedInitializer(
      std::chrono::milliseconds delay,
      std::function<void()> callback)
      : callback_(std::move(callback)) {
    thread_ = std::thread([this, delay]() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!cv_.wait_for(lock, delay, [this] { return stop_.load(); })) {
        callback_();
        called_ = true;
      }
    });
  }

  ~TestDelayedInitializer() {
    stop_ = true;
    cv_.notify_one();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  bool wasCalled() const {
    return called_.load();
  }

 private:
  std::function<void()> callback_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> called_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
};

} // namespace

TEST(DelayedInitializerTest, CallsCallbackAfterDelay) {
  std::atomic<bool> initialized{false};
  {
    TestDelayedInitializer init(
        std::chrono::milliseconds(50), [&]() { initialized = true; });
    EXPECT_FALSE(initialized.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(initialized.load());
  }
}

TEST(DelayedInitializerTest, CancelsOnEarlyDestruction) {
  std::atomic<bool> initialized{false};
  {
    TestDelayedInitializer init(
        std::chrono::milliseconds(500), [&]() { initialized = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  EXPECT_FALSE(initialized.load());
}

TEST(DelayedInitializerTest, DestructorDoesNotBlock) {
  auto start = std::chrono::steady_clock::now();
  {
    TestDelayedInitializer init(std::chrono::seconds(10), []() {});
  }
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));
}
