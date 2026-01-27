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

include(ExternalProject)

function(make_test test_file)
  get_filename_component(test_name "${test_file}" NAME_WE)
  set(lib_name "${test_name}Lib")
  add_library(${lib_name} SHARED XpuptiTestUtilities.cpp ${test_file})
  target_link_libraries(${lib_name} PRIVATE ${LINK_LIBRARIES})
  set_target_properties(${lib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
  install(TARGETS ${lib_name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  ExternalProject_Add(${test_name}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/compute
    CMAKE_ARGS -DPROJECT_NAME=${test_name}
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${SYCL_COMPILER}
    CMAKE_ARGS -DCMAKE_PARENT_BINARY_DIR=${CMAKE_BINARY_DIR}
    CMAKE_ARGS -DLINK_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/lib${lib_name}.so
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
    BUILD_ALWAYS TRUE
  )

  add_dependencies(${test_name} ${lib_name})
  add_dependencies(${test_name} gtest)
  add_dependencies(${test_name} gtest_main)
endfunction()

make_test(XpuptiScopeProfilerTest.cpp)
