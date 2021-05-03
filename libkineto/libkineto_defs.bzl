# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_libkineto_api_srcs():
    return [
        "src/ThreadUtil.cpp",
        "src/libkineto_api.cpp",
    ]

def get_libkineto_srcs(with_api = True):
    return [
        "src/AbstractConfig.cpp",
        "src/ActivityProfiler.cpp",
        "src/ActivityProfilerController.cpp",
        "src/ActivityProfilerProxy.cpp",
        "src/Config.cpp",
        "src/ConfigLoader.cpp",
        "src/CudaDeviceProperties.cpp",
        "src/CuptiActivityInterface.cpp",
        "src/CuptiEventInterface.cpp",
        "src/CuptiMetricInterface.cpp",
        "src/Demangle.cpp",
        "src/EventProfiler.cpp",
        "src/EventProfilerController.cpp",
        "src/GenericTraceActivity.cpp",
        "src/Logger.cpp",
        "src/WeakSymbols.cpp",
        "src/cupti_strings.cpp",
        "src/init.cpp",
        "src/output_csv.cpp",
        "src/output_json.cpp",
    ] + (get_libkineto_api_srcs() if with_api else [])

def get_libkineto_cpu_only_srcs(with_api = True):
    return [
        "src/AbstractConfig.cpp",
        "src/ActivityProfiler.cpp",
        "src/ActivityProfilerController.cpp",
        "src/ActivityProfilerProxy.cpp",
        "src/Config.cpp",
        "src/ConfigLoader.cpp",
        "src/CuptiActivityInterface.cpp",
        "src/Demangle.cpp",
        "src/GenericTraceActivity.cpp",
        "src/Logger.cpp",
        "src/init.cpp",
        "src/output_csv.cpp",
        "src/output_json.cpp",
    ] + (get_libkineto_api_srcs() if with_api else [])

def get_libkineto_public_headers():
    return [
        "include/ActivityProfilerInterface.h",
        "include/ActivityType.h",
        "include/ClientInterface.h",
        "include/GenericTraceActivity.h",
        "include/TraceActivity.h",
        "include/TraceSpan.h",
        "include/ThreadUtil.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
    ]
