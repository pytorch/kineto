// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DeviceType.h"

#include <fmt/format.h>

namespace libkineto {

struct DeviceTypeName {
  const char* name;
  DeviceType type;
};

static constexpr std::array<DeviceTypeName, deviceTypeCount + 1> map{{
    {"cpu", DeviceType::CPU},
    {"xpu", DeviceType::XPU},
    {"cuda", DeviceType::CUDA},
    {"ENUM_COUNT", DeviceType::ENUM_COUNT}
}};

static constexpr bool matchingOrder(int idx = 0) {
  return map[idx].type == DeviceType::ENUM_COUNT ||
    ((idx == (int) map[idx].type) && matchingOrder(idx + 1));
}
static_assert(matchingOrder(), "DeviceTypeName map is out of order");

const char* toString(DeviceType t) {
  return map[(int)t].name;
}

DeviceType toDeviceType(const std::string& str) {
  for (int i = 0; i < deviceTypeCount; i++) {
    if (str == map[i].name) {
      return map[i].type;
    }
  }
  throw std::invalid_argument(fmt::format("Invalid device type: {}", str));
}

const std::array<DeviceType, deviceTypeCount> deviceTypes() {
  std::array<DeviceType, deviceTypeCount> res;
  for (int i = 0; i < deviceTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}

const std::array<DeviceType, defaultDeviceTypeCount> defaultDeviceTypes() {
  std::array<DeviceType, defaultDeviceTypeCount> res;
  for (int i = 0; i < defaultDeviceTypeCount; i++) {
    res[i] = map[i].type;
  }
  return res;
}


} // namespace libkineto
