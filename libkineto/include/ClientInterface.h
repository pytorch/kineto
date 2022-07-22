// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace libkineto {

class ClientInterface {
 public:
  virtual ~ClientInterface() {}
  virtual void init() = 0;
  virtual void warmup(bool setupOpInputsCollection) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

} // namespace libkineto
