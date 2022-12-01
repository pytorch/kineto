/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <functional>

namespace KINETO_NAMESPACE {

class Config;

class IActivityProfilerThread {
 public:
  virtual ~IActivityProfilerThread() {}
  virtual void enqueue(
      std::function<void()> f,
      std::chrono::time_point<std::chrono::system_clock> when) = 0;

  virtual bool active() const = 0;

  virtual bool isCurrentThread() const = 0;
};

} // namespace KINETO_NAMESPACE
