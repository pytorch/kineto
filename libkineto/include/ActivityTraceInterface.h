// Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
// All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace libkineto {

struct ITraceActivity;

class ActivityTraceInterface {
 public:
  virtual ~ActivityTraceInterface() {}
  virtual const std::vector<const ITraceActivity*>* activities() {
    return nullptr;
  }
  virtual void save(const std::string& path) {}
};

} // namespace libkineto
