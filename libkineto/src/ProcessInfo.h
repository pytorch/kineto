/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace KINETO_NAMESPACE {

struct ProcessInfo {
  int32_t pid;
  const std::string name;
  const std::string label;
};

struct ThreadInfo {
  ThreadInfo(int32_t tid, const std::string name) :
    tid(tid), name(name) {}
  int32_t tid;
  const std::string name;
};


// Return a list of pids and process names for the current process
// and its parents.
std::vector<std::pair<int32_t, std::string>> pidCommandPairsOfAncestors();

} // namespace KINETO_NAMESPACE
