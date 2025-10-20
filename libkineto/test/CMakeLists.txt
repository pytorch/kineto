# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
    if((NOT SYCL_INCLUDE_DIR) OR (NOT SYCL_LIBRARY_DIR))
        include(FindSYCLToolkit.cmake)
        if(NOT SYCLTOOLKIT_FOUND)
            set(LIBKINETO_NOXPUPTI ON CACHE STRING "" FORCE PARENT_SCOPE)
            message(WARNING "XPU PTI has not built because ${SYCL_NOT_FOUND_MESSAGE}")
        endif()
    endif()
endif()

if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
    set(XPU_xpupti_LIBRARY ${SYCL_LIBRARY})
    list(APPEND XPU_xpupti_LIBRARY ${PTI_LIBRARY})
    set(XPU_xpupti_LIBRARY ${XPU_xpupti_LIBRARY} PARENT_SCOPE)
else()
    set(XPU_xpupti_LIBRARY "")
endif()

include_directories(${LIBKINETO_DIR})
link_libraries(fmt::fmt-header-only)
# ConfigTest
add_executable(ConfigTest ConfigTest.cpp)
target_link_libraries(ConfigTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_xpupti_LIBRARY})
gtest_discover_tests(ConfigTest)

if(NOT LIBKINETO_NOCUPTI)
# CuptiActivityProfilerTest
#[[
add_executable(CuptiActivityProfilerTest
    CuptiActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp)
target_link_libraries(CuptiActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api)
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
target_link_libraries(CuptiCallbackApiTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(CuptiCallbackApiTest)

# CuptiRangeProfilerApiTest
# add_executable(CuptiRangeProfilerApiTest CuptiRangeProfilerApiTest.cpp)
# target_link_libraries(CuptiRangeProfilerApiTest PRIVATE
#     gtest_main
#     kineto
#     CUDA::cupti
#     $<BUILD_INTERFACE:fmt::fmt-header-only>)
# Skipping due to SEGFault in 12.4
# Tracked here: https://github.com/pytorch/kineto/issues/949
# gtest_discover_tests(CuptiRangeProfilerApiTest)

# CuptiRangeProfilerConfigTest
add_executable(CuptiRangeProfilerConfigTest CuptiRangeProfilerConfigTest.cpp)
target_link_libraries(CuptiRangeProfilerConfigTest PRIVATE
    gtest_main
    kineto_base kineto_api
    $<BUILD_INTERFACE:fmt::fmt-header-only>)
gtest_discover_tests(CuptiRangeProfilerConfigTest)

# CuptiRangeProfilerTest
# add_executable(CuptiRangeProfilerTest CuptiRangeProfilerTest.cpp)
# target_link_libraries(CuptiRangeProfilerTest PRIVATE
#     gtest_main
#     kineto
#     CUDA::cupti
#     $<BUILD_INTERFACE:fmt::fmt-header-only>)
# Skipping due to SEGFault in 12.4
# Tracked here: https://github.com/pytorch/kineto/issues/949
# gtest_discover_tests(CuptiRangeProfilerTest)

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

if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
    #include(CheckCXXCompilerFlag)
    #check_cxx_compiler_flag("-fsycl" has_sycl)

    #if (NOT has_sycl)
        #message(WARNING "${PROJECT_NAME} requires a sycl compatible compiler")
        #return()
    #endif()

    # XpuptiScopeProfilerConfigTest
    add_executable(XpuptiScopeProfilerConfigTest XpuptiScopeProfilerConfigTest.cpp)
    target_link_libraries(XpuptiScopeProfilerConfigTest PRIVATE
        gtest_main
        kineto_base kineto_api
        $<BUILD_INTERFACE:fmt::fmt-header-only>
        "${XPU_xpupti_LIBRARY}")
    gtest_discover_tests(XpuptiScopeProfilerConfigTest)

    # XpuptiScopeProfilerTest
    add_executable(XpuptiScopeProfilerTest XpuptiScopeProfilerTest.cpp XpuptiScopeProfilerCompute.cpp)
    #target_compile_options(XpuptiScopeProfilerTest PUBLIC -fsycl)
    #target_link_options(XpuptiScopeProfilerTest PUBLIC -fsycl)
    target_link_libraries(XpuptiScopeProfilerTest PRIVATE
        gtest_main
        kineto_base kineto_api
        $<BUILD_INTERFACE:fmt::fmt-header-only>
        "${XPU_xpupti_LIBRARY}")
    gtest_discover_tests(XpuptiScopeProfilerTest)
endif()

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_link_libraries(LoggerObserverTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_xpupti_LIBRARY})
gtest_discover_tests(LoggerObserverTest)

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_link_libraries(PidInfoTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_xpupti_LIBRARY})
gtest_discover_tests(PidInfoTest)

# CuptiProfilerApiTest
# enable_language(CUDA)
# add_executable(CuptiProfilerApiTest CuptiProfilerApiTest.cu)
# target_link_libraries(CuptiProfilerApiTest PRIVATE
#     kineto
#     gtest_main)
#add_test(NAME CuptiProfilerApiTest_ COMMAND CuptiProfilerApiTest)
