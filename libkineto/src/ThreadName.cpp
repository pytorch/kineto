/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <pthread.h>

namespace KINETO_NAMESPACE {

static constexpr size_t kMaxThreadNameLength = 16;

bool setThreadName(const std::string& name) {
  return 0 == pthread_setname_np(pthread_self(), name.c_str());
}

std::string getThreadName(pthread_t id) {
  char buf[kMaxThreadNameLength];
  if (0 == pthread_getname_np(id, buf, sizeof(buf))) {
    return buf;
  }
  return "Unknown";
}

} // namespace KINETO_NAMESPACE
