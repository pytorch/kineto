/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <level_zero/ze_api.h>
#include <plugin/xpupti/XpuptiVersionLogger.h>
#include <pti/pti_view.h>
#include <sycl/version.hpp>
namespace KINETO_NAMESPACE {
namespace {

std::string getLevelZeroVersion() {
  uint32_t version = 0;
  auto result = zeInit(0);
  if (result == ZE_RESULT_SUCCESS) {
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);
    if (driverCount > 0) {
      ze_driver_handle_t driver;
      zeDriverGet(&driverCount, &driver);
      ze_api_version_t apiVersion;
      zeDriverGetApiVersion(driver, &apiVersion);

      return std::to_string(ZE_MAJOR_VERSION(apiVersion)) + "." +
          std::to_string(ZE_MINOR_VERSION(apiVersion));
    }
  }
  return "unknown";
}

} // namespace

void XpuVersionLogger::logAndRecordVersions() {
  std::string ptiVersion = "\"" + std::string(ptiVersionString()) + "\"";
  addVersionMetadata("pti_version", ptiVersion);

  addVersionMetadata("level_zero_version", getLevelZeroVersion());

  std::string syclCompilerVersion = std::to_string(__SYCL_COMPILER_VERSION);
  addVersionMetadata("sycl_compiler_version", syclCompilerVersion);
}

} // namespace KINETO_NAMESPACE
