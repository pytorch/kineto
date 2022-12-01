/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include "IActivityProfilerThread.h"

namespace KINETO_NAMESPACE {

class Config;
class IProfilerSession;
class ICompositeProfiler;

class ActivityProfilerThread : public IActivityProfilerThread {
 public:
  ActivityProfilerThread(
      ICompositeProfiler& profiler,
      const Config& cfg,
      std::promise<std::shared_ptr<IProfilerSession>> promise);
  virtual ~ActivityProfilerThread();

  bool active() const override {
    return active_;
  }

  void enqueue(
      std::function<void()> f,
      std::chrono::time_point<std::chrono::system_clock> when) override;

  bool isCurrentThread() const override;

 private:
  void start();
  std::pair<
      std::chrono::time_point<std::chrono::system_clock>,
      std::function<void()>>
  nextStep();

  ICompositeProfiler& profiler_;
  std::unique_ptr<Config> config_;
  std::promise<std::shared_ptr<IProfilerSession>> promise_;
  std::unique_ptr<std::thread> thread_;
  std::atomic_bool stopFlag_{false};
  std::atomic_bool active_{true};
  mutable std::mutex mutex_;
  std::condition_variable threadCondVar_;
  std::list<
      std::pair<
          std::chrono::time_point<std::chrono::system_clock>,
          std::function<void()>>> workQueue;
};

} // namespace KINETO_NAMESPACE
