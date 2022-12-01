/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <memory>

namespace libkineto {

class Config;
class IProfilerSession;

class IConfigHandler {
 public:
  virtual ~IConfigHandler() {}

  // Return true if handler is currently able to accept a new config.
  // May return false if e.g. profiler is busy or disabled.
  virtual bool canAcceptConfig() = 0;

  // Called for a new request.
  // Handler can apply the config immediately (if quick) or delegate to
  // a worker thread. Errors are reported to the profiler session.
  virtual std::future<std::shared_ptr<IProfilerSession>>
  acceptConfig(const Config& cfg) = 0;
};

} // namespace libkineto
