# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
set(CMAKE_CXX_STANDARD 17)

if(KINETO_BACKEND STREQUAL "xpu")
    set(XPU_XPUPTI_LIBRARY ${PTI_LIBRARY} ${SYCL_LIBRARY})
    add_subdirectory(xpupti)
else()
    set(XPU_XPUPTI_LIBRARY "")
endif()

include_directories(${LIBKINETO_DIR})
link_libraries(fmt::fmt-header-only)
# ApproximateClockTest
add_executable(ApproximateClockTest ApproximateClockTest.cpp)
target_link_libraries(ApproximateClockTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ApproximateClockTest)

# ConfigTest
add_executable(ConfigTest ConfigTest.cpp)
target_link_libraries(ConfigTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ConfigTest)

if(KINETO_BACKEND STREQUAL "cuda")
# CuptiActivityProfilerTest
add_executable(CuptiActivityProfilerTest
    CuptiActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp)
target_link_libraries(CuptiActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api
    nlohmann_json::nlohmann_json)
target_include_directories(CuptiActivityProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src"
    "${CUDA_SOURCE_DIR}/include"
    "${CUPTI_INCLUDE_DIR}")
gtest_discover_tests(CuptiActivityProfilerTest)
# CuptiCallbackApiTest
add_executable(CuptiCallbackApiTest CuptiCallbackApiTest.cpp)
target_link_libraries(CuptiCallbackApiTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(CuptiCallbackApiTest)

# CuptiRangeProfilerApiTest
add_executable(CuptiRangeProfilerApiTest CuptiRangeProfilerApiTest.cpp)
target_link_libraries(CuptiRangeProfilerApiTest PRIVATE
    gtest_main
    kineto_base kineto_api
    $<BUILD_INTERFACE:fmt::fmt-header-only>)
gtest_discover_tests(CuptiRangeProfilerApiTest)

# CuptiRangeProfilerConfigTest
add_executable(CuptiRangeProfilerConfigTest CuptiRangeProfilerConfigTest.cpp)
target_link_libraries(CuptiRangeProfilerConfigTest PRIVATE
    gtest_main
    kineto_base kineto_api
    $<BUILD_INTERFACE:fmt::fmt-header-only>)
gtest_discover_tests(CuptiRangeProfilerConfigTest)

# CuptiRangeProfilerTest
add_executable(CuptiRangeProfilerTest CuptiRangeProfilerTest.cpp)
target_link_libraries(CuptiRangeProfilerTest PRIVATE
    gtest_main
    kineto_base kineto_api
    $<BUILD_INTERFACE:fmt::fmt-header-only>)
gtest_discover_tests(CuptiRangeProfilerTest)

# CuptiStringsTest
add_executable(CuptiStringsTest CuptiStringsTest.cpp)
target_link_libraries(CuptiStringsTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(CuptiStringsTest)

# EventProfilerTest
add_executable(EventProfilerTest EventProfilerTest.cpp)
target_link_libraries(EventProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api)
gtest_discover_tests(EventProfilerTest)
endif()

if(KINETO_BACKEND STREQUAL "rocm")
# RocmActivityProfilerTest
add_executable(RocmActivityProfilerTest
    RocmActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp)
target_compile_definitions(RocmActivityProfilerTest PRIVATE
    "__HIP_PLATFORM_HCC__"
    "__HIP_PLATFORM_AMD__")
target_link_libraries(RocmActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api
    nlohmann_json::nlohmann_json)
target_include_directories(RocmActivityProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src"
    "${ROCM_INCLUDE_DIRS}"
    "${ROCTRACER_INCLUDE_DIR}")
gtest_discover_tests(RocmActivityProfilerTest)
endif()

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_link_libraries(LoggerObserverTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(LoggerObserverTest)

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_link_libraries(PidInfoTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(PidInfoTest)

# CuptiProfilerApiTest
# TODO: Re-enable once runtime crash in cuptiDeviceGetChipName is fixed.
# enable_language(CUDA)
# add_executable(CuptiProfilerApiTest CuptiProfilerApiTest.cu)
# target_link_libraries(CuptiProfilerApiTest PRIVATE
#     kineto_base kineto_api
#     CUDA::cuda_driver
#     gtest_main)
# add_test(NAME CuptiProfilerApiTest COMMAND CuptiProfilerApiTest)
