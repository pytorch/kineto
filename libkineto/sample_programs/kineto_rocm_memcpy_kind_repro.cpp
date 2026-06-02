/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

#include <folly/init/Init.h>
#include <libkineto.h>

namespace {

constexpr const char* kDefaultTracePath =
    "/tmp/kineto_rocm_memcpy_kind_repro.json";

struct Options {
  std::size_t bytes{1 << 20};
  int iterations{64};
  int printLimit{40};
  std::string tracePath{kDefaultTracePath};
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--bytes N] [--iterations N] [--print-limit N]"
               " [--trace-path PATH]\n\n"
            << "Runs a focused ROCm/Kineto memcpy-kind repro on AMD GPUs:\n"
            << "  1. hipMemcpyAsync(..., hipMemcpyHostToDevice, stream)\n"
            << "  2. hipMemcpyWithStream(..., hipMemcpyDeviceToHost, stream)\n"
            << "Then prints Kineto runtime/GPU memcpy activities and saves a "
               "Chrome trace JSON.\n";
  std::exit(2);
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto requireValue = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        usage(argv[0]);
      }
      return argv[++i];
    };

    if (arg == "--bytes") {
      options.bytes = std::stoull(requireValue("--bytes"));
    } else if (arg == "--iterations") {
      options.iterations = std::stoi(requireValue("--iterations"));
    } else if (arg == "--print-limit") {
      options.printLimit = std::stoi(requireValue("--print-limit"));
    } else if (arg == "--trace-path") {
      options.tracePath = requireValue("--trace-path");
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0]);
    }
  }

  if (options.bytes == 0 || options.iterations <= 0 || options.printLimit < 0) {
    throw std::invalid_argument(
        "--bytes and --iterations must be positive; --print-limit must be non-negative");
  }
  return options;
}

void checkHip(
    hipError_t status,
    const char* expression,
    const char* file,
    int line) {
  if (status == hipSuccess) {
    return;
  }
  std::cerr << "HIP error at " << file << ":" << line << " while running "
            << expression << ": " << hipGetErrorString(status) << "\n";
  std::exit(1);
}

#define CHECK_HIP(expr) checkHip((expr), #expr, __FILE__, __LINE__)

void fillHostBuffer(std::byte* buffer, std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i) {
    buffer[i] = static_cast<std::byte>(i & 0xff);
  }
}

void runMemcpyPair(
    void* deviceBuffer,
    void* hostInput,
    void* hostOutput,
    std::size_t bytes,
    hipStream_t stream) {
  CHECK_HIP(hipMemcpyAsync(
      deviceBuffer, hostInput, bytes, hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipMemcpyWithStream(
      hostOutput, deviceBuffer, bytes, hipMemcpyDeviceToHost, stream));
}

bool isRelevantMemcpyActivity(const libkineto::ITraceActivity& activity) {
  const auto type = activity.type();
  const auto name = activity.name();
  return type == libkineto::ActivityType::GPU_MEMCPY ||
      (type == libkineto::ActivityType::CUDA_RUNTIME &&
       name.find("Memcpy") != std::string::npos);
}

std::string extractKindForSummary(const std::string& metadata) {
  constexpr const char* marker = "\"kind\": ";
  const auto pos = metadata.find(marker);
  if (pos == std::string::npos) {
    return "kind=<missing>";
  }

  auto valueStart = pos + std::strlen(marker);
  while (valueStart < metadata.size() && metadata[valueStart] == ' ') {
    ++valueStart;
  }
  if (valueStart >= metadata.size()) {
    return "kind=<missing>";
  }
  if (metadata[valueStart] == '"') {
    const auto valueEnd = metadata.find('"', valueStart + 1);
    if (valueEnd != std::string::npos) {
      return "kind=" +
          metadata.substr(valueStart + 1, valueEnd - valueStart - 1);
    }
  }

  const auto valueEnd = metadata.find_first_of(",\n", valueStart);
  return "kind=" + metadata.substr(valueStart, valueEnd - valueStart);
}

void printRelevantActivity(const libkineto::ITraceActivity& activity) {
  const auto type = activity.type();
  std::cout << "KINETO_ACTIVITY"
            << " type=" << libkineto::toString(type) << " name=\""
            << activity.name() << "\" resource=" << activity.resourceId()
            << " correlation=" << activity.correlationId()
            << " duration_ns=" << activity.duration()
            << " metadata=" << activity.metadataJson() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  const Options options = parseOptions(argc, argv);

  int initArgc = 1;
  char* initArgvStorage[] = {argv[0], nullptr};
  char** initArgv = initArgvStorage;
  const folly::Init init(&initArgc, &initArgv);

  int deviceCount = 0;
  CHECK_HIP(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::cerr << "No HIP devices found. Run this on an AMD devgpu host.\n";
    return 1;
  }

  CHECK_HIP(hipSetDevice(0));
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  std::cout << "Using HIP device 0: " << props.name << "\n";
  std::cout << "Runtime source of truth:\n"
            << "  hipMemcpyAsync(..., kind=hipMemcpyHostToDevice/"
            << static_cast<int>(hipMemcpyHostToDevice) << ")\n"
            << "  hipMemcpyWithStream(..., kind=hipMemcpyDeviceToHost/"
            << static_cast<int>(hipMemcpyDeviceToHost) << ")\n";

  hipStream_t stream = nullptr;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  void* deviceBuffer = nullptr;
  void* hostInput = nullptr;
  void* hostOutput = nullptr;
  CHECK_HIP(hipMalloc(&deviceBuffer, options.bytes));
  CHECK_HIP(hipHostMalloc(&hostInput, options.bytes));
  CHECK_HIP(hipHostMalloc(&hostOutput, options.bytes));
  fillHostBuffer(static_cast<std::byte*>(hostInput), options.bytes);
  std::memset(hostOutput, 0, options.bytes);

  runMemcpyPair(deviceBuffer, hostInput, hostOutput, options.bytes, stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  libkineto_init(false, true);
  libkineto::api().initProfilerIfRegistered();

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::GPU_MEMCPY,
      libkineto::ActivityType::GPU_MEMSET,
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::EXTERNAL_CORRELATION,
      libkineto::ActivityType::CUDA_RUNTIME,
      libkineto::ActivityType::CUDA_DRIVER,
  };

  auto& profiler = libkineto::api().activityProfiler();
  profiler.prepareTrace(
      activityTypes,
      "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
      "ACTIVITIES_DURATION_MSECS=0\n");

  profiler.startTrace();
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  for (int i = 0; i < options.iterations; ++i) {
    runMemcpyPair(deviceBuffer, hostInput, hostOutput, options.bytes, stream);
  }
  CHECK_HIP(hipStreamSynchronize(stream));

  auto trace = profiler.stopTrace();
  std::cout << "Stopped Kineto trace with " << trace->activities()->size()
            << " activities\n";

  int relevantCount = 0;
  int printedCount = 0;
  std::map<std::string, int> memcpyActivityCounts;
  for (const auto& activity : *trace->activities()) {
    if (!isRelevantMemcpyActivity(*activity)) {
      continue;
    }
    ++relevantCount;
    const auto summaryKey = std::string(libkineto::toString(activity->type())) +
        " " + activity->name() + " " +
        extractKindForSummary(activity->metadataJson());
    ++memcpyActivityCounts[summaryKey];
    if (printedCount < options.printLimit) {
      printRelevantActivity(*activity);
      ++printedCount;
    }
  }

  std::cout << "Relevant memcpy activities: " << relevantCount << "\n";
  for (const auto& [summaryKey, count] : memcpyActivityCounts) {
    std::cout << "SUMMARY " << count << " x " << summaryKey << "\n";
  }

  trace->save(options.tracePath);
  std::cout << "Saved trace: " << options.tracePath << "\n";

  CHECK_HIP(hipHostFree(hostOutput));
  CHECK_HIP(hipHostFree(hostInput));
  CHECK_HIP(hipFree(deviceBuffer));
  CHECK_HIP(hipStreamDestroy(stream));
  return 0;
}
