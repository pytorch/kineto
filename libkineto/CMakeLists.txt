cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(kineto VERSION 0.1 LANGUAGES CXX C)

set(KINETO_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (default or shared) to build")
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

# Define file lists
get_filelist("get_libkineto_srcs()" LIBKINETO_SRCS)
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
set(LIBKINETO_API_SRCS "${LIBKINETO_SOURCE_DIR}/external_api.cpp")
set(FMT_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt/include")
set(CUPTI_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/cupti/include")
set(CUDA_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/cuda-9.2/include")

message(INFO " CUPTI_INCLUDE_DIR = ${CUPTI_INCLUDE_DIR}")

add_library(kineto_base OBJECT ${LIBKINETO_SRCS})
add_library(kineto_api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(kineto_base defs.bzl)

set_target_properties(kineto_base kineto_api PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden)

target_compile_options(kineto_base PRIVATE "-DKINETO_NAMESPACE=libkineto"
"-std=gnu++14")
target_compile_options(kineto_api PRIVATE "-std=gnu++14")

if(NOT TARGET fmt)
  #Download fmt from github if FMT_SOURCE_DIR is not specified.
  if(NOT DEFINED FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  #build fmt
  set(FMT_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_BINARY_DIR}/fmt")
  set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(kineto_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUPTI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>)

target_include_directories(kineto_api PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>)

if(KINETO_LIBRARY_TYPE STREQUAL "default")
  add_library(kineto
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

target_link_libraries(kineto $<BUILD_INTERFACE:fmt>)
add_dependencies(kineto fmt)

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
