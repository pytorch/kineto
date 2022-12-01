/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "ActivityProfilerThread.h"

namespace KINETO_NAMESPACE {

ActivityProfilerThread::ActivityProfilerThread(const Config& cfg) {
  thread_ = std::make_unique<std::thread>(cfg);
} // namespace KINETO_NAMESPACE

void ActivityProfilerThread::execute(std::function<void()> f) {

}

ActivityProfilerThread::scheduleTrace(const Config& cfg) {

}

