#[=======================================================================[.rst:
AIUConfig
-------

Library to verify AIU compatability of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``AIUTOOLKIT_FOUND``
  True if the system has the AIU library.
``AIU_COMPILER``
  AIU compiler executable.
``AIU_INCLUDE_DIR``
  Include directories needed to use AIU.
``AIU_LIBRARY``
  Library directories needed to use AIU.

#]=======================================================================]

set(AIU_EXECUTABLE_NAME "compile_graph")

set(AIUTOOLKIT_FOUND False)
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

# Set AIU_ROOT based on environment variables or fallback
if(DEFINED ENV{RUNTIME_FULL_INSTALL_DIR})
  set(AIU_ROOT $ENV{RUNTIME_FULL_INSTALL_DIR})
elseif(DEFINED ENV{SENDNN_DIR})
  set(AIU_ROOT $ENV{SENDNN_DIR})
else()
  execute_process(
    COMMAND which ${AIU_EXECUTABLE_NAME}
    OUTPUT_VARIABLE AIU_CMPLR_FULL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT EXISTS "${AIU_CMPLR_FULL_PATH}")
    message(WARNING "Cannot find RUNTIME_FULL_INSTALL_DIR or SENDNN_DIR, please set up the AIU environment.")
    return()
  endif()

  get_filename_component(AIU_BIN_DIR "${AIU_CMPLR_FULL_PATH}" DIRECTORY)
  set(AIU_ROOT ${AIU_BIN_DIR}/..) # Assume the compiler path gives us AIU root
endif()

# Set LIBAIUPTI_INSTALL_DIR based on environment variable or fallback
if(DEFINED ENV{LIBAIUPTI_INSTALL_DIR})
  set(LIBAIUPTI_INSTALL_DIR $ENV{LIBAIUPTI_INSTALL_DIR})
else()
  set(LIBAIUPTI_INSTALL_DIR ${AIU_ROOT})
endif()

# Find the AIU compiler
find_program(
  AIU_COMPILER
  NAMES ${AIU_EXECUTABLE_NAME}
  PATHS "${AIU_ROOT}/bin" "${AIU_ROOT}/bin64"
  NO_DEFAULT_PATH
)

# Verify the compiler was found
if(NOT AIU_COMPILER)
  set(AIUTOOLKIT_FOUND False)
  message(WARNING "AIU: Compiler not found.")
  return()
endif()

# Find the AIU include directory
find_path(
  AIU_INCLUDE_DIR
  NAMES "aiupti_activity.h"
  PATHS "${LIBAIUPTI_INSTALL_DIR}/include/libaiupti"
  NO_DEFAULT_PATH
)

# Find the AIU library directory
find_library(
  AIU_LIBRARY
  NAMES "aiupti"
  PATHS "${LIBAIUPTI_INSTALL_DIR}/lib" "${LIBAIUPTI_INSTALL_DIR}/lib64"
  NO_DEFAULT_PATH
)

# Check if all components were found
if((NOT AIU_INCLUDE_DIR) OR(NOT AIU_LIBRARY))
  set(AIUTOOLKIT_FOUND False)
  message(WARNING "AIU SDK is incomplete: include directory or library not found.")
  return()
endif()

# Success case
set(AIUTOOLKIT_FOUND True)
message(STATUS "AIU library found: ${AIU_LIBRARY}")
message(STATUS "AIU include directory: ${AIU_INCLUDE_DIR}")
message(STATUS "AIU compiler: ${AIU_COMPILER}")
