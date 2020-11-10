/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/types.h>
#include <string>
#include <utility>
#include <vector>

namespace KINETO_NAMESPACE {

struct ProcessInfo {
  pid_t pid;
  const std::string name;
  const std::string label;
};

struct ThreadInfo {
  int64_t tid;
  const std::string name;
};


// Return a list of pids and process names for the current process
// and its parents.
std::vector<std::pair<pid_t, std::string>> pidCommandPairsOfAncestors();

} // namespace KINETO_NAMESPACE
