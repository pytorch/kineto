# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from libkineto_defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('libkineto_defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(kineto VERSION 0.1 LANGUAGES CXX C)

set(KINETO_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (default, static or shared) to build")
set_property(CACHE KINETO_LIBRARY_TYPE PROPERTY STRINGS default shared)
option(KINETO_BUILD_TESTS "Build kineto unit tests" ON)

set(LIBKINETO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBKINETO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBKINETO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBKINETO_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if (NOT CUDA_SOURCE_DIR)
    set(CUDA_SOURCE_DIR "$ENV{CUDA_SOURCE_DIR}")
    message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

if (NOT ROCM_SOURCE_DIR)
    set(ROCM_SOURCE_DIR "$ENV{ROCM_SOURCE_DIR}")
    message(INFO " ROCM_SOURCE_DIR = ${ROCM_SOURCE_DIR}")
endif()

# Set LIBKINETO_NOCUPTI to explicitly disable CUPTI
# Otherwise, CUPTI is disabled if not found
IF (NOT CUDA_SOURCE_DIR OR NOT CUPTI_INCLUDE_DIR OR NOT CUDA_cupti_LIBRARY)
    set(LIBKINETO_NOCUPTI ON CACHE BOOL "" FORCE)
endif()

IF (NOT ROCM_SOURCE_DIR AND NOT ROCTRACER_INCLUDE_DIR)
    set(LIBKINETO_NOROCTRACER ON CACHE BOOL "" FORCE)
endif()

# Define file lists
if (LIBKINETO_NOCUPTI AND LIBKINETO_NOROCTRACER)
    get_filelist("get_libkineto_cpu_only_srcs(with_api=False)" LIBKINETO_SRCS)
    message(INFO " CUPTI unavailable or disabled - not building GPU profilers")
elseif(NOT LIBKINETO_NOROCTRACER)
    get_filelist("get_libkineto_roctracer_srcs(with_api=False)" LIBKINETO_SRCS)
    message(INFO " Building with roctracer")
else()
    get_filelist("get_libkineto_cupti_srcs(with_api=False)" LIBKINETO_SRCS)
endif()
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
get_filelist("get_libkineto_api_srcs()" LIBKINETO_API_SRCS)

add_library(kineto_base OBJECT ${LIBKINETO_SRCS})
add_library(kineto_api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on libkineto_defs.bzl
add_custom_target(libkineto_defs.bzl DEPENDS libkineto_defs.bzl)
add_dependencies(kineto_base libkineto_defs.bzl)

set_target_properties(kineto_base kineto_api PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

set(KINETO_COMPILE_OPTIONS "-DKINETO_NAMESPACE=libkineto")
list(APPEND KINETO_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
list(APPEND KINETO_COMPILE_OPTIONS "-DENABLE_IPC_FABRIC")
if(NOT MSVC)
  list(APPEND KINETO_COMPILE_OPTIONS "-std=c++17")
else()
  list(APPEND KINETO_COMPILE_OPTIONS "/std:c++17")
  list(APPEND KINETO_COMPILE_OPTIONS "-DWIN32_LEAN_AND_MEAN")
  list(APPEND KINETO_COMPILE_OPTIONS "-DNOGDI")
endif()
if (NOT LIBKINETO_NOCUPTI)
  list(APPEND KINETO_COMPILE_OPTIONS "-DHAS_CUPTI")
endif()
if (CUDA_nvperf_host_LIBRARY)
  list(APPEND KINETO_COMPILE_OPTIONS "-DUSE_CUPTI_RANGE_PROFILER")
endif()
if (NOT LIBKINETO_NOROCTRACER)
    target_compile_options(kineto_base PRIVATE "-DHAS_ROCTRACER")
    target_compile_options(kineto_base PRIVATE "-D__HIP_PLATFORM_HCC__")
    target_compile_options(kineto_base PRIVATE "-D__HIP_PLATFORM_AMD__")
endif()

target_compile_options(kineto_base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_options(kineto_api PRIVATE "${KINETO_COMPILE_OPTIONS}")

if(NOT TARGET fmt)
  if(NOT FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  # Build FMT.
  # FMT and some other libraries use BUILD_SHARED_LIBS to control
  # the library type.
  # Save and restore the value after configuring FMT
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
  set(FMT_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_BINARY_DIR}/fmt")
  set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
message(STATUS "Kineto: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")
if (NOT CUPTI_INCLUDE_DIR)
    set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/extras/CUPTI/include")
endif()
if (NOT CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS "${CUDA_SOURCE_DIR}/include")
endif()
if (NOT ROCTRACER_INCLUDE_DIR)
    set(ROCTRACER_INCLUDE_DIR "${ROCM_SOURCE_DIR}/include/roctracer")
endif()
if (NOT ROCM_INCLUDE_DIRS)
    set(ROCM_INCLUDE_DIRS "${ROCM_SOURCE_DIR}/include")
endif()

set(DYNOLOG_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/dynolog/")
set(IPCFABRIC_INCLUDE_DIR "${DYNOLOG_INCLUDE_DIR}/dynolog/src/ipcfabric/")

message(INFO " CUPTI_INCLUDE_DIR = ${CUPTI_INCLUDE_DIR}")
message(INFO " ROCTRACER_INCLUDE_DIR = ${ROCTRACER_INCLUDE_DIR}")
message(INFO " DYNOLOG_INCLUDE_DIR = ${DYNOLOG_INCLUDE_DIR}")
message(INFO " IPCFABRIC_INCLUDE_DIR = ${IPCFABRIC_INCLUDE_DIR}")

add_subdirectory("${IPCFABRIC_INCLUDE_DIR}")
target_link_libraries(kineto_base PRIVATE dynolog_ipcfabric_lib)

target_include_directories(kineto_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>
      $<BUILD_INTERFACE:${DYNOLOG_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${IPCFABRIC_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUPTI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${ROCTRACER_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${ROCM_INCLUDE_DIRS}>)

target_include_directories(kineto_api PUBLIC
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>)

if(KINETO_LIBRARY_TYPE STREQUAL "default")
  add_library(kineto
    $<TARGET_OBJECTS:kineto_base>
    $<TARGET_OBJECTS:kineto_api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "static")
  add_library(kineto STATIC
    $<TARGET_OBJECTS:kineto_base>
    $<TARGET_OBJECTS:kineto_api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "shared")
  add_library(kineto SHARED
    $<TARGET_OBJECTS:kineto_base>)
  set_property(TARGET kineto_base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(kineto PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
else()
  message(FATAL_ERROR "Unsupported library type ${KINETO_LIBRARY_TYPE}")
endif()

if(NOT LIBKINETO_NOROCTRACER)
  find_library(ROCTRACER_LIBRARY NAMES libroctracer64.so HINTS /opt/rocm/lib)
  target_link_libraries(kineto "${ROCTRACER_LIBRARY}")
  find_library(KINETO_HIP_LIBRARY NAMES libamdhip64.so HINTS /opt/rocm/lib)
  target_link_libraries(kineto "${KINETO_HIP_LIBRARY}")
endif()

if(NOT LIBKINETO_NOCUPTI)
  target_link_libraries(kineto "${CUDA_cupti_LIBRARY}")
endif()
if(CUDA_nvperf_host_LIBRARY)
  target_link_libraries(kineto "${CUDA_nvperf_host_LIBRARY}")
endif()
target_link_libraries(kineto $<BUILD_INTERFACE:fmt::fmt-header-only>)
add_dependencies(kineto fmt::fmt-header-only)

install(TARGETS kineto EXPORT kinetoLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBKINETO_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kineto")

install(EXPORT kinetoLibraryConfig DESTINATION share/cmake/kineto
  FILE kinetoLibraryConfig.cmake)

if(KINETO_BUILD_TESTS)
  add_subdirectory(test)
endif()
