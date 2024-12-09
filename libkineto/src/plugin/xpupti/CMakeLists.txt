# xpupti: XPU implementation for kineto profiler
# outputs:
#  SYCL_INCLUDE_DIR   --  SYCL include dir
#  SYCL_LIBRARY       --  SYCL library file
#  XPU_xpupti_LIBRARY --  XPUPTI dependencies lib
#  XPUPTI_BUILD_FLAG  --  XPUPTI build flags

if((NOT SYCL_INCLUDE_DIR) OR (NOT SYCL_LIBRARY_DIR))
    include(FindSYCLToolkit.cmake)
    if(NOT SYCLTOOLKIT_FOUND)
        set(LIBKINETO_NOXPUPTI ON CACHE STRING "" FORCE PARENT_SCOPE)
        message(WARNING "XPU PTI has not built because ${SYCL_NOT_FOUND_MESSAGE}")
        return()
    endif()
endif()

message(INFO " SYCL_INCLUDE_DIR = ${SYCL_INCLUDE_DIR}")
message(INFO " SYCL_LIBRARY = ${SYCL_LIBRARY}")

set(XPUPTI_INCLUDE_DIR ${SYCL_INCLUDE_DIR})

# find xpupti sdk
find_package(Pti REQUIRED)
if(TARGET Pti::pti_view)
    message(INFO " Found XPUPTI")

    set(pti_view_names)
    # For Linux, due to it has soft link original file name is fine.
    list(APPEND pti_view_names pti_view)
    # For Windows, file name to be appended with version number.
    foreach(ver_major RANGE 0 1)
        foreach(ver_minor RANGE 0 19)
        list(APPEND pti_view_names pti_view-${ver_major}-${ver_minor})
        endforeach()        
    endforeach()    

    get_target_property(PTI_INCLUDE_DIR Pti::pti_view INTERFACE_INCLUDE_DIRECTORIES)
    find_library(PTI_VIEW_LIBRARY NAMES ${pti_view_names} PATHS "${PTI_INCLUDE_DIR}/../lib")
    set(PTI_LIBRARY ${PTI_VIEW_LIBRARY} CACHE STRING "Imported PTI library.")
    set(PTI_INCLUDE_DIR ${PTI_INCLUDE_DIR} CACHE STRING "PTI include directory.")

    # find dependent lib
    set(XPU_xpupti_LIBRARY ${SYCL_LIBRARY})
    list(APPEND XPU_xpupti_LIBRARY ${PTI_LIBRARY})
    set(XPU_xpupti_LIBRARY ${XPU_xpupti_LIBRARY} PARENT_SCOPE)

    # find dependent include
    list(APPEND XPUPTI_INCLUDE_DIR ${PTI_INCLUDE_DIR})
    set(XPUPTI_INCLUDE_DIR ${XPUPTI_INCLUDE_DIR} PARENT_SCOPE)

    set(XPUPTI_BUILD_FLAG "-DHAS_XPUPTI" PARENT_SCOPE)

    message(INFO " XPU_xpupti_LIBRARY = ${XPU_xpupti_LIBRARY}")
    message(INFO " XPUPTI_INCLUDE_DIR = ${XPUPTI_INCLUDE_DIR}")
    message(INFO " XPUPTI_BUILD_FLAG = ${XPUPTI_BUILD_FLAG}")
else()
    set(LIBKINETO_NOXPUPTI ON CACHE STRING "" FORCE PARENT_SCOPE)
    set(XPU_xpupti_LIBRARY PARENT_SCOPE)
    set(XPUPTI_BUILD_FLAG PARENT_SCOPE)
    set(XPUPTI_INCLUDE_DIR PARENT_SCOPE)
    message(WARNING " Could not find XPUPTI for building kineto")
    return()
endif()
