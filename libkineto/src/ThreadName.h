/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <pthread.h>

namespace KINETO_NAMESPACE {

bool setThreadName(const std::string& name);
std::string getThreadName(pthread_t id);

} // namespace KINETO_NAMESPACE
