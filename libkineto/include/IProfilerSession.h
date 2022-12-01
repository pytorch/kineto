/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace libkineto {

class IProfilerSession {

 public:
  virtual ~IProfilerSession() {};

  // Clients must lock this to protect against concurrent access
  // before calling any functions that read or write state.
  virtual std::mutex& mutex() = 0;

  // returns errors that occurred during profiling
  virtual std::vector<std::string> errors() = 0;
};

} // namespace libkineto

