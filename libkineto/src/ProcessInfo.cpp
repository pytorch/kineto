/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ProcessInfo.h"

#include <fmt/format.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "Logger.h"

static const std::string kChronosJobIDEnvVar = "CHRONOS_JOB_INSTANCE_ID";

namespace KINETO_NAMESPACE {

// Max number of parent pids to collect, just for extra safeguarding.
constexpr int kMaxParentPids = 10;

// Return a pair of <parent_pid, command_of_current_pid>
static std::pair<pid_t, std::string> parentPidAndCommand(pid_t pid) {
  FILE* statfile = fopen(fmt::format("/proc/{}/stat", pid).c_str(), "r");
  if (statfile == nullptr) {
    return std::make_pair(0, "");
  }
  pid_t parent_pid;
  char* command = nullptr;
  int scanned = fscanf(statfile, "%*d (%m[^)]) %*c %d", &command, &parent_pid);
  fclose(statfile);
  VLOG(2) << " Current PID: " << pid << " Command: " << command
          << " Parent PID: " << parent_pid;
  std::pair<pid_t, std::string> ret;
  if (scanned == 2) {
    ret = std::make_pair(parent_pid, std::string(command));
  } else {
    LOG(ERROR) << "Failed to parse /proc/" << pid << "/stat";
    ret = std::make_pair(0, "");
  }

  // The 'm' character in the format tells fscanf to allocate memory
  // for the parsed string, which we need to free here.
  free(command);
  return ret;
}

std::vector<std::pair<pid_t, std::string>> pidCommandPairsOfAncestors() {
  std::vector<std::pair<pid_t, std::string>> pairs;
  pairs.reserve(kMaxParentPids + 1);
  pid_t curr_pid = getpid();
  for (int i = 0; i <= kMaxParentPids && curr_pid > 1; i++) {
    std::pair<pid_t, std::string> ppid_and_comm = parentPidAndCommand(curr_pid);
    pairs.push_back(std::make_pair(curr_pid, ppid_and_comm.second));
    curr_pid = ppid_and_comm.first;
  }
  return pairs;
}

} // namespace KINETO_NAMESPACE
