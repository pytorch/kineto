/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

namespace libkineto {

struct TraceActivity;

class ActivityTraceInterface {
 public:
  virtual ~ActivityTraceInterface() {}
  virtual const std::vector<std::unique_ptr<TraceActivity>>* activities() {
    return nullptr;
  }
  virtual void save(const std::string& path) {}
};

} // namespace libkineto
