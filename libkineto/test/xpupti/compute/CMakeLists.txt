# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(XpuptiScopeProfilerTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" has_sycl)

if(NOT has_sycl)
  message(WARNING "${PROJECT_NAME} requires a sycl compatible compiler")
  return()
endif()

function(find_libraries)
  foreach(lib IN LISTS ARGN)
    find_library(_FIND_LIB
      NAMES ${lib}
      PATHS ${CMAKE_PARENT_BINARY_DIR}
      PATH_SUFFIXES lib
      NO_DEFAULT_PATH
    )

    if(NOT _FIND_LIB)
      message(FATAL_ERROR "Library ${lib} not found")
    endif()

    set(${lib}_FOR_XPU_PATH "${_FIND_LIB}" PARENT_SCOPE)
    unset(_FIND_LIB CACHE)
  endforeach()
endfunction()

find_libraries(gtest gtest_main)

add_executable(XpuptiScopeProfilerTest XpuptiScopeProfilerCompute.cpp)
target_compile_options(XpuptiScopeProfilerTest PRIVATE -fsycl)
target_link_options(XpuptiScopeProfilerTest PRIVATE -fsycl)

target_link_libraries(XpuptiScopeProfilerTest PRIVATE
  ${gtest_main_FOR_XPU_PATH}
  ${gtest_FOR_XPU_PATH}
  ${LINK_LIBRARY}
)

set_target_properties(XpuptiScopeProfilerTest PROPERTIES INSTALL_RPATH "$ORIGIN")

gtest_discover_tests(XpuptiScopeProfilerTest)

install(TARGETS XpuptiScopeProfilerTest DESTINATION .)
