/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <pthread.h>
#include <sys/types.h>
#include <string>
#include <unistd.h>

#include "Logger.h"

namespace KINETO_NAMESPACE {

static constexpr size_t kMaxThreadNameLength = 16;

bool setThreadName(const std::string& name) {
  return 0 == pthread_setname_np(pthread_self(), name.c_str());
}

std::string getThreadName(pid_t tid) {
  char buf[kMaxThreadNameLength] = "Unknown";
  std::string filename = fmt::format("/proc/{}/task/{}/comm", getpid(), tid);
  FILE* comm_file = fopen(filename.c_str(), "r");
  if (comm_file) {
    size_t len = fread(buf, 1, kMaxThreadNameLength, comm_file);
    fclose(comm_file);
    // Remove newline
    if (len > 0) {
      buf[len - 1] = '\0';
    }
  } else {
    LOG(WARNING) << "Failed to open " << filename;
  }
  return buf;
}

} // namespace KINETO_NAMESPACE
