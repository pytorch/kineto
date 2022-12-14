# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library (dynolog_ipcfabric_lib INTERFACE)
target_include_directories(dynolog_ipcfabric_lib
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(dynolog_ipcfabric_lib
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
