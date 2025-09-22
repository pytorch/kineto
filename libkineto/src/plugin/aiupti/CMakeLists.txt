# aiupti: AIU implementation for kineto profiler
# outputs:
# AIU_INCLUDE_DIR   --  AIU include dir
# AIU_LIBRARY       --  AIU library file
# AIU_aiupti_LIBRARY --  AIUPTI dependencies lib
# AIUPTI_BUILD_FLAGS  --  AIUPTI build flags

# Clean up old/bad definitions
unset(LIBKINETO_NOAIUPTI CACHE)

set(LIBKINETO_NOAIUPTI OFF CACHE BOOL "Disable AIUPTI" FORCE)

if((NOT AIU_INCLUDE_DIR) OR (NOT AIU_LIBRARY_DIR))
    include(FindAIUToolkit.cmake)
    if(NOT AIUTOOLKIT_FOUND)
        set(LIBKINETO_NOAIUPTI ON CACHE BOOL "Disable AIUPTI" FORCE)
        message(WARNING "AIU PTI has not built because ${AIU_NOT_FOUND_MESSAGE}")
        return()
    endif()
endif()

message(INFO " AIU_INCLUDE_DIR = ${AIU_INCLUDE_DIR}")
message(INFO " AIU_LIBRARY = ${AIU_LIBRARY}")

# Setup variables
set(AIUPTI_INCLUDE_DIR ${AIU_INCLUDE_DIR} CACHE STRING "" FORCE)
set(AIU_aiupti_LIBRARY ${AIU_LIBRARY} CACHE STRING "" FORCE)
set(AIUPTI_BUILD_FLAGS "-DHAS_AIUPTI" CACHE STRING "" FORCE)

if(NOT DISPLAY_TRACE_IN_MS)
    list(APPEND AIUPTI_BUILD_FLAGS "-DDISPLAY_TRACE_IN_NS")
endif()

# Propagate variables to the parent scope
set(AIUPTI_INCLUDE_DIR ${AIUPTI_INCLUDE_DIR} PARENT_SCOPE)
set(AIU_aiupti_LIBRARY ${AIU_aiupti_LIBRARY} PARENT_SCOPE)
set(AIUPTI_BUILD_FLAGS ${AIUPTI_BUILD_FLAGS} PARENT_SCOPE)
