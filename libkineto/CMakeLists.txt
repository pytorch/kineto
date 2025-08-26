# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(kineto VERSION 0.1 LANGUAGES CXX C)

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from libkineto_defs.bzl file
find_package(Python3 COMPONENTS Interpreter)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "exec(open('libkineto_defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

set(KINETO_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (default, static or shared) to build")
set_property(CACHE KINETO_LIBRARY_TYPE PROPERTY STRINGS default shared static)
option(KINETO_BUILD_TESTS "Build kineto unit tests" ON)

set(LIBKINETO_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBKINETO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBKINETO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBKINETO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBKINETO_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#We should default to a Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if(DEFINED CUDA_SOURCE_DIR AND NOT DEFINED CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${CUDA_SOURCE_DIR}")
  message(STATUS " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

if(NOT ROCM_SOURCE_DIR)
  set(ROCM_SOURCE_DIR "$ENV{ROCM_SOURCE_DIR}")
  message(STATUS " ROCM_SOURCE_DIR = ${ROCM_SOURCE_DIR}")
endif()

# Set LIBKINETO_NOCUPTI to explicitly disable CUPTI
# Otherwise, CUPTI is disabled if not found
option(LIBKINETO_NOCUPTI "Disable CUPTI" OFF)
option(USE_CUPTI_SO "Use CUPTI as a shared library" ON)

find_package(CUDAToolkit)
if(NOT LIBKINETO_NOCUPTI)
  if(NOT CUPTI_INCLUDE_DIR)
    find_path(CUPTI_INCLUDE_DIR cupti.h PATHS
      "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include"
      "${CUDAToolkit_LIBRARY_ROOT}"
      "${CUDAToolkit_LIBRARY_ROOT}/include"
      NO_DEFAULT_PATH)
  endif()

  if(USE_CUPTI_SO)
    if(NOT CUDA_cupti_LIBRARY)
      find_library(CUDA_cupti_LIBRARY cupti PATHS
        "${CUDAToolkit_LIBRARY_ROOT}"
        "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
        "${CUDAToolkit_LIBRARY_ROOT}/lib"
        "${CUDAToolkit_LIBRARY_ROOT}/lib64"
        NO_DEFAULT_PATH)
    endif()
    set(cupti_LIBRARY ${CUDA_cupti_LIBRARY})
  else()
    if(NOT CUDA_cupti_static_LIBRARY)
      find_library(CUDA_cupti_static_LIBRARY cupti_static PATHS
        "${CUDAToolkit_LIBRARY_ROOT}"
        "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
        "${CUDAToolkit_LIBRARY_ROOT}/lib"
        "${CUDAToolkit_LIBRARY_ROOT}/lib64"
        NO_DEFAULT_PATH)
    endif()
    set(cupti_LIBRARY ${CUDA_cupti_static_LIBRARY})
  endif()

  if(cupti_LIBRARY AND CUPTI_INCLUDE_DIR)
    message(STATUS "  CUPTI_INCLUDE_DIR = ${CUPTI_INCLUDE_DIR}")
    message(STATUS "  CUDA_cupti_LIBRARY = ${CUDA_cupti_LIBRARY}")
    message(STATUS "  CUDA_cupti_static_LIBRARY = ${CUDA_cupti_static_LIBRARY}")
    message(STATUS "Found CUPTI")
    if(USE_CUPTI_SO)
      set(cupti_target CUDA::cupti)
    else()
      set(cupti_target CUDA::cupti_static)
    endif()
    if(NOT TARGET ${cupti_target})
      add_library(${cupti_target} INTERFACE IMPORTED)
      target_link_libraries(${cupti_target} INTERFACE "${cupti_LIBRARY}")
    endif()
    target_include_directories(${cupti_target} INTERFACE "${CUPTI_INCLUDE_DIR}")

    if(NOT USE_CUPTI_SO)
      include(CMakePushCheckState)
      include(CheckSourceRuns)
      cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} $<LINK_LIBRARY:WHOLE_ARCHIVE,${cupti_LIBRARY}>)

      check_source_runs(CXX "#include <stdexcept>
  int main() {
  try {
    throw std::runtime_error(\"error\");
  } catch (...) {
    return 0;
  }
  return 1;
  }" EXCEPTIONS_WORK)
      if(NOT EXCEPTIONS_WORK)
        message(FATAL_ERROR
          "Detected that statically linking against CUPTI causes exceptions to stop working. "
          "See https://github.com/pytorch/pytorch/issues/57744 for more details. ")
      endif()
      cmake_pop_check_state()
    endif()

  else()
    set(LIBKINETO_NOCUPTI ON CACHE BOOL "" FORCE)
    message(STATUS "Could not find CUPTI library")
  endif()
endif()
if(NOT TARGET CUDA::nvperf_host)
  find_library(CUDA_NVPERF_HOST_LIB_PATH nvperf_host PATHS
        "${CUDAToolkit_LIBRARY_ROOT}/lib/x64"
        "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
        "${CUDAToolkit_LIBRARY_ROOT}/lib"
        "${CUDAToolkit_LIBRARY_ROOT}/lib64"
        NO_DEFAULT_PATH)
  if(CUDA_NVPERF_HOST_LIB_PATH)
    message(STATUS "Found NVPERF: ${CUDA_NVPERF_HOST_LIB_PATH}")
    if(WIN32)
      add_library(CUDA::nvperf_host SHARED IMPORTED)
      set_target_properties(CUDA::nvperf_host PROPERTIES
        IMPORTED_IMPLIB "${CUDA_NVPERF_HOST_LIB_PATH}"
      )
    else()
      add_library(CUDA::nvperf_host INTERFACE IMPORTED)
      target_link_libraries(CUDA::nvperf_host INTERFACE "${CUDA_NVPERF_HOST_LIB_PATH}")
    endif()
  endif()
endif()

if(NOT ROCM_SOURCE_DIR AND NOT ROCTRACER_INCLUDE_DIR)
  set(LIBKINETO_NOROCTRACER ON CACHE BOOL "" FORCE)
endif()

if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
  add_subdirectory(src/plugin/xpupti)
else()
  set(LIBKINETO_NOXPUPTI ON)
endif()

# Define file lists
if(LIBKINETO_NOCUPTI AND LIBKINETO_NOROCTRACER AND LIBKINETO_NOXPUPTI)
  get_filelist("get_libkineto_cpu_only_srcs(with_api=False)" LIBKINETO_SRCS)
  message(STATUS " CUPTI unavailable or disabled - not building GPU profilers")
else()
  if(NOT LIBKINETO_NOROCTRACER)
    get_filelist("get_libkineto_roctracer_srcs(with_api=False)" LIBKINETO_roc_SRCS)
    message(STATUS " Building with roctracer")
  elseif(NOT LIBKINETO_NOCUPTI)
    get_filelist("get_libkineto_cupti_srcs(with_api=False)" LIBKINETO_cuda_SRCS)
  endif()
  if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
    get_filelist("get_libkineto_xpupti_srcs(with_api=False)" LIBKINETO_xpu_SRCS)
    message(STATUS " Building with xpupti")
  endif()
  set(LIBKINETO_SRCS ${LIBKINETO_roc_SRCS} ${LIBKINETO_xpu_SRCS} ${LIBKINETO_cuda_SRCS})
endif()
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
get_filelist("get_libkineto_api_srcs()" LIBKINETO_API_SRCS)

add_library(kineto_base OBJECT ${LIBKINETO_SRCS})
add_library(kineto_api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on libkineto_defs.bzl
add_custom_target(libkineto_defs.bzl DEPENDS libkineto_defs.bzl)
add_dependencies(kineto_base libkineto_defs.bzl)

set(KINETO_DEFINITIONS "KINETO_NAMESPACE=libkineto")
list(APPEND KINETO_DEFINITIONS "ENABLE_IPC_FABRIC")
set(KINETO_COMPILE_OPTIONS)
if(MSVC)
  list(APPEND KINETO_COMPILE_OPTIONS "/utf-8")
endif()
if(NOT LIBKINETO_NOCUPTI)
  list(APPEND KINETO_DEFINITIONS "HAS_CUPTI")
endif()
if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
  list(APPEND KINETO_COMPILE_OPTIONS ${XPUPTI_BUILD_FLAG})
endif()
if(TARGET CUDA::nvperf_host)
  list(APPEND KINETO_DEFINITIONS "USE_CUPTI_RANGE_PROFILER")
endif()
if(NOT LIBKINETO_NOROCTRACER)
  list(APPEND KINETO_DEFINITIONS "HAS_ROCTRACER")
  target_compile_definitions(kineto_base PRIVATE "__HIP_PLATFORM_HCC__")
  target_compile_definitions(kineto_base PRIVATE "__HIP_PLATFORM_AMD__")
endif()

target_compile_definitions(kineto_base PUBLIC "${KINETO_DEFINITIONS}")
target_compile_options(kineto_base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_definitions(kineto_api PUBLIC "${KINETO_DEFINITIONS}")
target_compile_options(kineto_api PRIVATE "${KINETO_COMPILE_OPTIONS}")

if(NOT ROCTRACER_INCLUDE_DIR)
  set(ROCTRACER_INCLUDE_DIR "${ROCM_SOURCE_DIR}/include/roctracer")
endif()
if(NOT ROCM_INCLUDE_DIRS)
  set(ROCM_INCLUDE_DIRS "${ROCM_SOURCE_DIR}/include")
endif()

set(DYNOLOG_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/dynolog/")
set(IPCFABRIC_INCLUDE_DIR "${DYNOLOG_INCLUDE_DIR}/dynolog/src/ipcfabric/")
add_subdirectory("${IPCFABRIC_INCLUDE_DIR}")

if(NOT TARGET fmt::fmt-header-only)
  if(NOT FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  # Build FMT.
  # FMT and some other libraries use BUILD_SHARED_LIBS to control
  # the library type.
  # Save and restore the value after configuring FMT
  set(FMT_INSTALL OFF)
  add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_BINARY_DIR}/fmt")
  message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
endif()

message(STATUS " ROCTRACER_INCLUDE_DIR = ${ROCTRACER_INCLUDE_DIR}")
message(STATUS " DYNOLOG_INCLUDE_DIR = ${DYNOLOG_INCLUDE_DIR}")
message(STATUS " IPCFABRIC_INCLUDE_DIR = ${IPCFABRIC_INCLUDE_DIR}")

target_link_libraries(kineto_base PRIVATE dynolog_ipcfabric_lib)

target_include_directories(kineto_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>
      $<BUILD_INTERFACE:${DYNOLOG_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${IPCFABRIC_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${ROCTRACER_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${ROCM_INCLUDE_DIRS}>)

if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
  target_include_directories(kineto_base PUBLIC ${XPUPTI_INCLUDE_DIR})
endif()
target_link_libraries(kineto_base PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

target_include_directories(kineto_api PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>)
target_link_libraries(kineto_api PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

if(KINETO_LIBRARY_TYPE STREQUAL "static")
  set(BUILD_SHARED_LIBS OFF)
elseif(KINETO_LIBRARY_TYPE STREQUAL "shared")
  set(BUILD_SHARED_LIBS ON)
elseif(NOT KINETO_LIBRARY_TYPE STREQUAL "default")
  message(FATAL_ERROR "Unsupported library type ${KINETO_LIBRARY_TYPE}")
endif()

add_library(kineto $<TARGET_OBJECTS:kineto_base> $<TARGET_OBJECTS:kineto_api>)
if(BUILD_SHARED_LIBS)
  set_property(TARGET kineto_base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET kineto_api PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(kineto PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

set_target_properties(kineto kineto_base kineto_api PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

target_include_directories(kineto PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>)

if(NOT LIBKINETO_NOROCTRACER)
  find_library(ROCTRACER_LIBRARY NAMES libroctracer64.so HINTS
    ${ROCM_SOURCE_DIR}/lib)
  target_link_libraries(kineto "${ROCTRACER_LIBRARY}")
  find_library(KINETO_HIP_LIBRARY NAMES libamdhip64.so HINTS
    ${ROCM_SOURCE_DIR}/lib)
  target_link_libraries(kineto "${KINETO_HIP_LIBRARY}")
endif()

if(NOT LIBKINETO_NOCUPTI)
  target_link_libraries(kineto PUBLIC ${cupti_target} CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(kineto_base PUBLIC ${cupti_target} CUDA::cudart CUDA::cuda_driver)
endif()
if(TARGET CUDA::nvperf_host)
  target_link_libraries(kineto_base PUBLIC CUDA::nvperf_host)
endif()
if(DEFINED LIBKINETO_NOXPUPTI AND NOT LIBKINETO_NOXPUPTI)
  target_link_libraries(kineto "${XPU_xpupti_LIBRARY}")
endif()
target_compile_definitions(kineto PUBLIC "${KINETO_DEFINITIONS}")

if(KINETO_BUILD_TESTS)
  if(NOT TARGET gtest)
    set(INSTALL_GTEST OFF)
    set(TMP_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory("${LIBKINETO_THIRDPARTY_DIR}/googletest")
    set(BUILD_SHARED_LIBS "${TMP_BUILD_SHARED_LIBS}")
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

install(TARGETS kineto
  EXPORT kinetoLibraryConfig
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBKINETO_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kineto")

install(EXPORT kinetoLibraryConfig DESTINATION share/cmake/kineto
  FILE kinetoLibraryConfig.cmake)
