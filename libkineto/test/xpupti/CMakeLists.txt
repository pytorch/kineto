# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 17)

# Do not link gtest into the shared library to avoid duplicate gtest globals
# (gtest is statically linked into the executable; the shared library resolves
#  gtest symbols from the executable at runtime via --export-dynamic).
set(COMMON_LINK_LIBRARIES
  kineto_base
  kineto_api
  $<BUILD_INTERFACE:fmt::fmt-header-only>
  ${SYCL_LIBRARY}
  ${PTI_LIBRARY}
)

set(LINK_LIBRARIES gtest_main ${COMMON_LINK_LIBRARIES})

add_executable(XpuptiScopeProfilerConfigTest XpuptiScopeProfilerConfigTest.cpp)
target_link_libraries(XpuptiScopeProfilerConfigTest PRIVATE ${LINK_LIBRARIES})
gtest_add_tests(TARGET XpuptiScopeProfilerConfigTest)

add_executable(XpuptiActivityHandlersTest XpuptiActivityHandlersTest.cpp)
target_link_libraries(XpuptiActivityHandlersTest PRIVATE ${LINK_LIBRARIES})
gtest_add_tests(TARGET XpuptiActivityHandlersTest)

include(ExternalProject)

function(make_sycl_test test_file)
  get_filename_component(test_name "${test_file}" NAME_WE)
  set(lib_name "${test_name}Lib")
  add_library(${lib_name} SHARED XpuptiTestUtilities.cpp ${test_file})

  target_link_libraries(${lib_name} PRIVATE ${COMMON_LINK_LIBRARIES})
  target_include_directories(${lib_name} PRIVATE
    $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>)

  set_target_properties(${lib_name} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
  # On Windows, DLLs must resolve all symbols at link time (unlike Linux .so).
  # ComputeOnXpu is defined in the compute executable (ExternalProject) which
  # links back to this DLL. /FORCE:UNRESOLVED allows the DLL to build with
  # this symbol unresolved. The linker will still report LNK2019 as an error
  # but /FORCE overrides it (followed by LNK4088 warning) — this is expected.
  if(WIN32)
    target_link_options(${lib_name} PRIVATE "LINKER:/FORCE:UNRESOLVED")
  endif()

  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
  install(TARGETS ${lib_name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  set(_link_lib_search_dir ${CMAKE_CURRENT_BINARY_DIR})

  # Build the common cmake args for the inner ExternalProject (single-config Ninja).
  set(_ep_cmake_args
    -DPROJECT_NAME=${test_name}
    -DCMAKE_CXX_COMPILER=${SYCL_COMPILER}
    -DCMAKE_PARENT_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DLINK_LIBRARY_NAME=${lib_name}
    -DLINK_LIBRARY_DIR=${_link_lib_search_dir}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
  )
  if(WIN32)
    get_filename_component(_sycl_compiler_dir "${SYCL_COMPILER}" DIRECTORY)
    get_filename_component(_sycl_root "${_sycl_compiler_dir}" DIRECTORY)
    list(APPEND _ep_cmake_args
      "-DCMAKE_EXE_LINKER_FLAGS=/Qoption,link,/LIBPATH:\"${_sycl_root}/lib\""
    )
  endif()

  # - Multi-config (Visual Studio, Ninja Multi-Config): uses CONFIGURE_COMMAND
  #   with $<CONFIG> since CMAKE_ARGS doesn't support generator expressions.
  #   Ninja is forced as a single-config inner generator.
  # - Single-config on Windows: forces Ninja for SYCL/ICX compatibility.
  # - Single-config on Linux: forwards the outer generator (Ninja, Makefiles, etc.)
  #   via CMAKE_GENERATOR.
  get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_is_multi_config)
    ExternalProject_Add(${test_name}
      SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/compute
      CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -G Ninja
        -S <SOURCE_DIR>
        -B <BINARY_DIR>
        -DCMAKE_BUILD_TYPE=$<CONFIG>
        ${_ep_cmake_args}
      BUILD_ALWAYS TRUE
    )
  else()
    set(_selected_generator ${CMAKE_GENERATOR})
    if(WIN32)
      set(_selected_generator Ninja)
    endif()
    ExternalProject_Add(${test_name}
      SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/compute
      CMAKE_GENERATOR ${_selected_generator}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        ${_ep_cmake_args}
      BUILD_ALWAYS TRUE
    )
  endif()

  add_dependencies(${test_name} ${lib_name})
  add_dependencies(${test_name} gtest)
  add_dependencies(${test_name} gtest_main)
endfunction()

make_sycl_test(XpuptiProfilerTest.cpp)

