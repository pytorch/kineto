// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <string>

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
