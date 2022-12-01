/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityProfilerThread.h"
#include "Config.h"
#include "ICompositeProfiler.h"
#include "time_since_epoch.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

ActivityProfilerThread::ActivityProfilerThread(
    ICompositeProfiler& profiler,
    const Config& cfg,
    std::promise<std::shared_ptr<IProfilerSession>> promise) :
    profiler_(profiler), config_(cfg.clone()), promise_(std::move(promise)) {
  thread_ = std::make_unique<std::thread>(&ActivityProfilerThread::start, this);
}

ActivityProfilerThread::~ActivityProfilerThread() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stopFlag_ = true;
    threadCondVar_.notify_one();
  }
  thread_->join();
  LOG(INFO) << "Thread done";
}

void ActivityProfilerThread::start() {
  LOG(INFO) << "Thread started";
  auto session = profiler_.configure(*config_);
  LOG(INFO) << "Session: " << (void*) session.get() << " status=" << (int) session->status();
  auto& profiler = profiler_;
  const std::string& url = config_->activitiesLogUrl();
  enqueue(
      [&profiler, &session](){profiler.start(*session);},
      config_->requestTimestamp());
  enqueue(
      [&profiler, &session, &url](){
          profiler.stop(*session);
          LOG(INFO) << "Logging to " << url;
          auto logger = profiler.loggerFactory().makeLogger(url);
          LOG(INFO) << "Logging";
          session->log(*logger);
      },
      config_->requestTimestamp() + config_->activitiesDuration());

  auto now = system_clock::now();
  auto step = nextStep();
  LOG(INFO) << "Entering runloop, status=" << (int) session->status();
  while (session->status() != TraceStatus::ERROR && step.second) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      threadCondVar_.wait_until(lock, step.first);
    }
    if (stopFlag_) {
      break;
    }
    now = system_clock::now();
    if (now >= step.first) {
      LOG(INFO) << "Executing step for t=" << timeSinceEpoch(step.first);
      VLOG(1) << "Executing step for t=" << timeSinceEpoch(step.first);
      step.second();
      step = nextStep();
    }
  }
  active_ = false;
  promise_.set_value(session);
  LOG(INFO) << "Thread exiting";
}

std::pair<time_point<system_clock>, std::function<void()>>
ActivityProfilerThread::nextStep() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (workQueue.empty()) {
    return {time_point<system_clock>(), nullptr};
  } else {
    auto step = workQueue.front();
    workQueue.pop_front();
    return step;
  }
}

void ActivityProfilerThread::enqueue(
    std::function<void()> f, time_point<system_clock> when) {
  bool notify = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = workQueue.begin();
    for (; it != workQueue.end(); ++it) {
      if (it->first > when) {
        break;
      }
    }
    it = workQueue.insert(it, std::pair(when, f));
    // Notify thread of new work item if it's the next to be executed
    // and it's enqueued from another thread
    notify = it == workQueue.begin() &&
        std::this_thread::get_id() != thread_->get_id();
  }
  if (notify) {
    threadCondVar_.notify_one();
  }
}

bool ActivityProfilerThread::isCurrentThread() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return thread_ && thread_->get_id() == std::this_thread::get_id();
}

} // namespace KINETO_NAMESPACE
