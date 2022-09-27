// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <string>

namespace libkineto {

// Note : All device types are not enabled by default. Please add them
// at correct position in the enum
enum class DeviceType {
    // Device types enabled by default
    CPU = 0, // cpu backend 
    XPU,    // xpu backend
    CUDA,   // cuda backend, /w cupti

    ENUM_COUNT, // This is to add buffer and not used for any profiling logic. Add your new type before it.
};

const char* toString(DeviceType t);
DeviceType toDeviceType(const std::string& str);

// Return an array of all device types except COUNT
constexpr int deviceTypeCount = (int)DeviceType::ENUM_COUNT;
constexpr int defaultDeviceTypeCount = (int)DeviceType::ENUM_COUNT;
const std::array<DeviceType, deviceTypeCount> deviceTypes();
const std::array<DeviceType, defaultDeviceTypeCount> defaultDeviceTypes();

} // namespace libkineto
