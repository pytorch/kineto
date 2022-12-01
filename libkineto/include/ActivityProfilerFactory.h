/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace KINETO_NAMESPACE {

class IActivityProfiler;

class ActivityProfilerFactory {
 public:

  using Creator = std::function<std::unique_ptr<IActivityProfiler>()>;

  void addCreator(const std::string& type, Creator creator) {
    creators_[type] = creator;
  }

  std::unique_ptr<IActivityProfiler> create(const std::string& type) {
    const auto& it = creators_.find(type);
    return it == creators_.end() ? nullptr : it->second();
  }

 private:
  std::map<std::string, Creator> creators_;
};

} // namespace KINETO_NAMESPACE
