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
