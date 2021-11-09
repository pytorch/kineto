# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_libkineto_api_srcs():
    return [
        "src/ThreadUtil.cpp",
        "src/libkineto_api.cpp",
    ]

def get_libkineto_cupti_srcs(with_api = True):
    return [
        "src/CudaDeviceProperties.cpp",
        "src/CuptiActivityApi.cpp",
        "src/CuptiActivityPlatform.cpp",
        "src/CuptiEventApi.cpp",
        "src/CuptiMetricApi.cpp",
        "src/Demangle.cpp",
        "src/EventProfiler.cpp",
        "src/EventProfilerController.cpp",
        "src/WeakSymbols.cpp",
        "src/cupti_strings.cpp",
    ] + (get_libkineto_cpu_only_srcs(with_api))

def get_libkineto_roctracer_srcs(with_api = True):
    return [
        "src/RoctracerActivityApi.cpp",
    ] + (get_libkineto_cpu_only_srcs(with_api))

def get_libkineto_cpu_only_srcs(with_api = True):
    return [
        "src/AbstractConfig.cpp",
        "src/CuptiActivityProfiler.cpp",
        "src/ActivityProfilerController.cpp",
        "src/ActivityProfilerProxy.cpp",
        "src/ActivityType.cpp",
        "src/Config.cpp",
        "src/ConfigLoader.cpp",
        "src/CuptiActivityApi.cpp",
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
        "include/GenericTraceActivity.h",
        "include/IActivityProfiler.h",
        "include/TraceSpan.h",
        "include/ThreadUtil.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
    ]

# kineto code should be updated to not have to
# suppress these warnings.
KINETO_COMPILER_FLAGS = [
    "-fexceptions",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
]
