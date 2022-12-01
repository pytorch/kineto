/*
 * Copyright (c) Kineto Contributors
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace libkineto {

struct DeviceInfo {
  DeviceInfo(int64_t id, const std::string& name, const std::string& label) :
    id(id), name(name), label(label) {}
  int64_t id;
  const std::string name;
  const std::string label;
};

struct ResourceInfo {
  ResourceInfo(
      int64_t deviceId,
      int64_t id,
      int64_t sortIndex,
      const std::string& name) :
      id(id), sortIndex(sortIndex), deviceId(deviceId), name(name) {}
  int64_t id;
  int64_t sortIndex;
  int64_t deviceId;
  const std::string name;
};

} // namespace libkineto
