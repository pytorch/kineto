# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_libkineto_api_srcs():
    return [
        "src/libkineto_api.cpp",
        "utils/ThreadUtil.cpp",
    ]

def get_libkineto_cupti_srcs(with_api = True):
    return [
        "src/CuptiActivityApi.cpp",
        "src/CuptiActivityPlatform.cpp",
        "src/CuptiCallbackApi.cpp",
        "src/CuptiEventApi.cpp",
        "src/CuptiMetricApi.cpp",
        "src/CuptiRangeProfiler.cpp",
        "src/CuptiRangeProfilerApi.cpp",
        "src/CuptiRangeProfilerConfig.cpp",
        "src/CuptiNvPerfMetric.cpp",
        "src/EventProfiler.cpp",
        "src/EventProfilerController.cpp",
        "utils/CudaDeviceProperties.cpp",
        "utils/CudaUtil.cpp",
        "utils/WeakSymbols.cpp",
        "utils/cupti_strings.cpp",
    ] + (get_libkineto_cpu_only_srcs(with_api))

def get_libkineto_roctracer_srcs(with_api = True):
    return [
        "src/RoctracerActivityApi.cpp",
        "src/RoctracerLogger.cpp",
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
        "src/DaemonConfigLoader.cpp",
        "src/GenericTraceActivity.cpp",
        "src/ILoggerObserver.cpp",
        "src/IpcFabricConfigClient.cpp",
        "src/Logger.cpp",
        "src/LoggingAPI.cpp",
        "src/init.cpp",
        "src/output_csv.cpp",
        "src/output_json.cpp",
        "utils/Demangle.cpp",
    ] + (get_libkineto_api_srcs() if with_api else [])

def get_libkineto_public_headers():
    return [
        "include/AbstractConfig.h",
        "include/ActivityProfilerInterface.h",
        "include/ActivityTraceInterface.h",
        "include/ActivityType.h",
        "include/Config.h",
        "include/ClientInterface.h",
        "include/GenericTraceActivity.h",
        "include/IActivityProfiler.h",
        "include/ILoggerObserver.h",
        "include/ITraceActivity.h",
        "include/TraceSpan.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
        "utils/Demangle.h",
        "utils/ThreadUtil.h",
    ]

# kineto code should be updated to not have to
# suppress these warnings.
KINETO_COMPILER_FLAGS = [
    "-fexceptions",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
]
