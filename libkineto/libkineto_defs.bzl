# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_libkineto_srcs():
    return [
        "src/ActivityProfiler.cpp",
        "src/ActivityProfilerController.cpp",
        "src/ActivityProfilerProxy.cpp",
        "src/Config.cpp",
        "src/ConfigLoader.cpp",
        "src/CuptiActivityInterface.cpp",
        "src/CuptiEventInterface.cpp",
        "src/CuptiMetricInterface.cpp",
        "src/Demangle.cpp",
        "src/EventProfiler.cpp",
        "src/EventProfilerController.cpp",
        "src/Logger.cpp",
        "src/ProcessInfo.cpp",
        "src/ThreadName.cpp",
        "src/cupti_strings.cpp",
        "src/init.cpp",
        "src/libkineto_api.cpp",
        "src/output_csv.cpp",
        "src/output_json.cpp",
    ]

def get_libkineto_public_headers():
    return [
        "include/ActivityProfilerInterface.h",
        "include/ActivityType.h",
        "include/ClientInterface.h",
        "include/TraceActivity.h",
        "include/TraceSpan.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
    ]
