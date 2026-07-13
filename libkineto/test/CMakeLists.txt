# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
set(CMAKE_CXX_STANDARD 20)

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

# ActivityProfilerControllerTest
# Not supported on Windows, uses unix "unistd.h"
if(NOT WIN32)
    add_executable(ActivityProfilerControllerTest
        ActivityProfilerControllerTest.cpp
        TestUtils.cpp)
    target_link_libraries(ActivityProfilerControllerTest PRIVATE
        gtest_main
        kineto_base kineto_api
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(ActivityProfilerControllerTest)

    # AsyncActivityProfilerHandlerTest
    add_executable(AsyncActivityProfilerHandlerTest
        AsyncActivityProfilerHandlerTest.cpp
        TestUtils.cpp)
    target_link_libraries(AsyncActivityProfilerHandlerTest PRIVATE
        gtest_main
        kineto_base kineto_api
        nlohmann_json::nlohmann_json
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(AsyncActivityProfilerHandlerTest)
endif()

if(KINETO_BACKEND STREQUAL "cuda")
# CuptiActivityProfilerTest
add_executable(CuptiActivityProfilerTest
    CuptiActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp
    MockCpuActivityBuffer.cpp
    TestUtils.cpp)
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
    MockActivitySubProfiler.cpp
    MockCpuActivityBuffer.cpp
    TestUtils.cpp)
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
    "${ROCM_INCLUDE_DIRS}")
gtest_discover_tests(RocmActivityProfilerTest)
endif()

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_link_libraries(LoggerObserverTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(LoggerObserverTest)

# RegisterLoggerFactoryTest
add_executable(RegisterLoggerFactoryTest RegisterLoggerFactoryTest.cpp)
target_link_libraries(RegisterLoggerFactoryTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(RegisterLoggerFactoryTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(RegisterLoggerFactoryTest)

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_link_libraries(PidInfoTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(PidInfoTest)

# TypedMetadataTest
add_executable(TypedMetadataTest TypedMetadataTest.cpp)
target_link_libraries(TypedMetadataTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(TypedMetadataTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(TypedMetadataTest)

# GenericTraceActivityMetadataTest
add_executable(GenericTraceActivityMetadataTest
    GenericTraceActivityMetadataTest.cpp)
target_link_libraries(GenericTraceActivityMetadataTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(GenericTraceActivityMetadataTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(GenericTraceActivityMetadataTest)

# OutputJsonTest
add_executable(OutputJsonTest OutputJsonTest.cpp)
target_link_libraries(OutputJsonTest PRIVATE
    gtest_main
    kineto_base kineto_api
    nlohmann_json::nlohmann_json
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(OutputJsonTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(OutputJsonTest)
