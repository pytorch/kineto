# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 17)

set(LINK_LIBRARIES
  gtest_main
  kineto_base
  kineto_api
  $<BUILD_INTERFACE:fmt::fmt-header-only>
  ${SYCL_LIBRARY}
  ${PTI_LIBRARY}
)

add_executable(XpuptiScopeProfilerConfigTest XpuptiScopeProfilerConfigTest.cpp)
target_link_libraries(XpuptiScopeProfilerConfigTest PRIVATE ${LINK_LIBRARIES})
gtest_discover_tests(XpuptiScopeProfilerConfigTest)

add_library(XpuptiScopeProfilerTestLib SHARED XpuptiScopeProfilerTest.cpp)
target_link_libraries(XpuptiScopeProfilerTestLib PRIVATE ${LINK_LIBRARIES})
set_target_properties(XpuptiScopeProfilerTestLib PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
install(TARGETS XpuptiScopeProfilerTestLib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

include(ExternalProject)

ExternalProject_Add(XpuptiScopeProfilerTest
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/compute
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${SYCL_COMPILER}
  CMAKE_ARGS -DCMAKE_PARENT_BINARY_DIR=${CMAKE_BINARY_DIR}
  CMAKE_ARGS -DLINK_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/libXpuptiScopeProfilerTestLib.so
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
  BUILD_ALWAYS TRUE
)

add_dependencies(XpuptiScopeProfilerTest XpuptiScopeProfilerTestLib)
add_dependencies(XpuptiScopeProfilerTest gtest)
add_dependencies(XpuptiScopeProfilerTest gtest_main)
