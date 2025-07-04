# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
set(CMAKE_CXX_STANDARD 17)

# ConfigTest
add_executable(ConfigTest ConfigTest.cpp)
target_compile_options(ConfigTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(ConfigTest PRIVATE
    gtest_main
    kineto)
gtest_discover_tests(ConfigTest)

# CuptiActivityProfilerTest
#[[
add_executable(CuptiActivityProfilerTest
    CuptiActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp)
target_compile_options(CuptiActivityProfilerTest PRIVATE
    "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto
    "${CUDA_cudart_LIBRARY}")
target_include_directories(CuptiActivityProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src"
    "${CUDA_SOURCE_DIR}/include"
    "${CUPTI_INCLUDE_DIR}")
gtest_discover_tests(CuptiActivityProfilerTest)
]]
# CuptiCallbackApiTest
add_executable(CuptiCallbackApiTest CuptiCallbackApiTest.cpp)
target_compile_options(CuptiCallbackApiTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiCallbackApiTest PRIVATE
    gtest_main
    kineto)
gtest_discover_tests(CuptiCallbackApiTest)

# CuptiRangeProfilerApiTest
add_executable(CuptiRangeProfilerApiTest CuptiRangeProfilerApiTest.cpp)
target_compile_options(CuptiRangeProfilerApiTest PRIVATE
    "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerApiTest PRIVATE
    gtest_main
    kineto)
# Skipping due to SEGFault in 12.4
# Tracked here: https://github.com/pytorch/kineto/issues/949
# gtest_discover_tests(CuptiRangeProfilerApiTest)

# CuptiRangeProfilerConfigTest
add_executable(CuptiRangeProfilerConfigTest CuptiRangeProfilerConfigTest.cpp)
target_compile_options(CuptiRangeProfilerConfigTest PRIVATE
    "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerConfigTest PRIVATE
    gtest_main
    kineto)
gtest_discover_tests(CuptiRangeProfilerConfigTest)

# CuptiRangeProfilerTest
add_executable(CuptiRangeProfilerTest CuptiRangeProfilerTest.cpp)
target_compile_options(CuptiRangeProfilerTest PRIVATE
    "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerTest PRIVATE
    gtest_main
    kineto)
# Skipping due to SEGFault in 12.4
# Tracked here: https://github.com/pytorch/kineto/issues/949
# gtest_discover_tests(CuptiRangeProfilerTest)

# CuptiStringsTest
add_executable(CuptiStringsTest CuptiStringsTest.cpp)
target_compile_options(CuptiStringsTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiStringsTest PRIVATE
    gtest_main
    kineto)
gtest_discover_tests(CuptiStringsTest)

# EventProfilerTest
add_executable(EventProfilerTest EventProfilerTest.cpp)
target_compile_options(EventProfilerTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(EventProfilerTest PRIVATE
    gtest_main
    gmock
    kineto)
gtest_discover_tests(EventProfilerTest)

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_compile_options(LoggerObserverTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(LoggerObserverTest PRIVATE
    gtest_main
    kineto)
target_include_directories(LoggerObserverTest PRIVATE
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(LoggerObserverTest)

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_compile_options(PidInfoTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(PidInfoTest PRIVATE
    gtest_main
    kineto)
gtest_discover_tests(PidInfoTest)

# CuptiProfilerApiTest
# enable_language(CUDA)
# add_executable(CuptiProfilerApiTest CuptiProfilerApiTest.cu)
# target_compile_options(CuptiProfilerApiTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
# target_link_libraries(CuptiProfilerApiTest PRIVATE
#     kineto
#     gtest_main)
#add_test(NAME CuptiProfilerApiTest_ COMMAND CuptiProfilerApiTest)
