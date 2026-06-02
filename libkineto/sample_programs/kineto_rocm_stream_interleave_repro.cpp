/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kineto/libkineto/sample_programs/kineto_rocm_stream_interleave_repro_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <folly/init/Init.h>
#include <libkineto.h>

namespace {

constexpr const char* kDefaultTracePath =
    "/tmp/kineto_rocm_stream_interleave_repro.json";

struct Options {
  int workers{8};
  int launches{32};
  int elements{1 << 18};
  int kernelIterations{512};
  int printLimit{40};
  std::string tracePath{kDefaultTracePath};
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--workers N] [--launches N] [--elements N]"
               " [--kernel-iters N] [--print-limit N] [--trace-path PATH]\n\n"
            << "Runs a focused ROCm/Kineto stream-interleave repro on AMD "
               "GPUs. Each host worker owns one HIP stream and launches an "
               "identifiable kernel family on that stream.\n";
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

    if (arg == "--workers") {
      options.workers = std::stoi(requireValue("--workers"));
    } else if (arg == "--launches") {
      options.launches = std::stoi(requireValue("--launches"));
    } else if (arg == "--elements") {
      options.elements = std::stoi(requireValue("--elements"));
    } else if (arg == "--kernel-iters") {
      options.kernelIterations = std::stoi(requireValue("--kernel-iters"));
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

  if (options.workers <= 0 ||
      options.workers > kineto::samples::kIssueCMaxWorkers ||
      options.launches <= 0 || options.elements <= 0 ||
      options.kernelIterations <= 0 || options.printLimit < 0) {
    throw std::invalid_argument(
        "--workers must be in [1, 16]; --launches, --elements, and --kernel-iters must be positive; --print-limit must be non-negative");
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

std::string streamSetToString(const std::set<int64_t>& streams) {
  std::ostringstream out;
  bool first = true;
  for (const auto stream : streams) {
    if (!first) {
      out << ",";
    }
    out << stream;
    first = false;
  }
  return out.str();
}

bool isIssueCKernel(const libkineto::ITraceActivity& activity) {
  return activity.type() == libkineto::ActivityType::CONCURRENT_KERNEL &&
      activity.name().find("kineto_issue_c_worker_kernel") != std::string::npos;
}

void printKernelActivity(const libkineto::ITraceActivity& activity) {
  std::cout << "KINETO_KERNEL"
            << " name=\"" << activity.name() << "\""
            << " stream=" << activity.resourceId()
            << " correlation=" << activity.correlationId()
            << " duration_ns=" << activity.duration()
            << " metadata=" << activity.metadataJson() << "\n";
}

void runWorkerLaunches(
    int workerId,
    float* buffer,
    hipStream_t stream,
    const Options& options,
    std::atomic<int>& readyWorkers,
    std::atomic<bool>& start,
    std::string& error) {
  readyWorkers.fetch_add(1, std::memory_order_release);
  while (!start.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  for (int i = 0; i < options.launches; ++i) {
    const auto status = kineto::samples::launchIssueCWorkerKernel(
        workerId, buffer, options.elements, options.kernelIterations, stream);
    if (status != hipSuccess) {
      error = hipGetErrorString(status);
      return;
    }
  }

  const auto status = hipStreamSynchronize(stream);
  if (status != hipSuccess) {
    error = hipGetErrorString(status);
  }
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
  std::cout << "Issue C source-of-truth workload:\n"
            << "  workers=" << options.workers << "\n"
            << "  one host thread and one HIP stream per worker\n"
            << "  launches_per_worker=" << options.launches << "\n";

  std::vector<hipStream_t> streams(options.workers, nullptr);
  std::vector<float*> buffers(options.workers, nullptr);
  for (int worker = 0; worker < options.workers; ++worker) {
    CHECK_HIP(hipStreamCreateWithFlags(&streams[worker], hipStreamNonBlocking));
    std::cout << "WORKER_STREAM worker=" << worker
              << " hip_stream=" << streams[worker] << "\n";
    CHECK_HIP(hipMalloc(&buffers[worker], options.elements * sizeof(float)));
    CHECK_HIP(hipMemsetAsync(
        buffers[worker], 0, options.elements * sizeof(float), streams[worker]));
    CHECK_HIP(
        kineto::samples::launchIssueCWorkerKernel(
            worker,
            buffers[worker],
            options.elements,
            options.kernelIterations,
            streams[worker]));
    CHECK_HIP(hipStreamSynchronize(streams[worker]));
  }

  libkineto_init(false, true);
  libkineto::api().initProfilerIfRegistered();

  std::set<libkineto::ActivityType> activityTypes = {
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

  std::atomic<int> readyWorkers{0};
  std::atomic<bool> start{false};
  std::vector<std::string> workerErrors(options.workers);
  std::vector<std::thread> threads;
  threads.reserve(options.workers);
  for (int worker = 0; worker < options.workers; ++worker) {
    threads.emplace_back(
        runWorkerLaunches,
        worker,
        buffers[worker],
        streams[worker],
        std::cref(options),
        std::ref(readyWorkers),
        std::ref(start),
        std::ref(workerErrors[worker]));
  }

  while (readyWorkers.load(std::memory_order_acquire) < options.workers) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }

  for (int worker = 0; worker < options.workers; ++worker) {
    if (!workerErrors[worker].empty()) {
      std::cerr << "Worker " << worker << " failed: " << workerErrors[worker]
                << "\n";
      return 1;
    }
  }

  auto trace = profiler.stopTrace();
  std::cout << "Stopped Kineto trace with " << trace->activities()->size()
            << " activities\n";

  int kernelCount = 0;
  int printedCount = 0;
  std::map<int64_t, int> kernelsByStream;
  std::map<std::string, int> kernelsByName;
  std::map<std::string, std::set<int64_t>> streamsByName;
  for (const auto& activity : *trace->activities()) {
    if (!isIssueCKernel(*activity)) {
      continue;
    }
    ++kernelCount;
    ++kernelsByStream[activity->resourceId()];
    ++kernelsByName[activity->name()];
    streamsByName[activity->name()].insert(activity->resourceId());
    if (printedCount < options.printLimit) {
      printKernelActivity(*activity);
      ++printedCount;
    }
  }

  std::cout << "Issue C kernel activities: " << kernelCount << "\n";
  std::cout << "Issue C distinct kernel streams: " << kernelsByStream.size()
            << "\n";
  for (const auto& [stream, count] : kernelsByStream) {
    std::cout << "SUMMARY_STREAM stream=" << stream << " kernels=" << count
              << "\n";
  }
  for (const auto& [name, count] : kernelsByName) {
    std::cout << "SUMMARY_KERNEL name=\"" << name << "\" count=" << count
              << " streams=" << streamSetToString(streamsByName[name]) << "\n";
  }

  if (kernelCount != options.workers * options.launches) {
    std::cout << "ISSUE_C_WARNING expected_kernel_count="
              << options.workers * options.launches
              << " observed_kernel_count=" << kernelCount << "\n";
  }
  if (static_cast<int>(kernelsByStream.size()) < options.workers) {
    std::cout << "ISSUE_C_WARNING expected_at_least_streams=" << options.workers
              << " observed_streams=" << kernelsByStream.size() << "\n";
  }
  for (const auto& [name, streams] : streamsByName) {
    if (streams.size() > 1) {
      std::cout << "ISSUE_C_WARNING kernel_name_spread name=\"" << name
                << "\" streams=" << streamSetToString(streams) << "\n";
    }
  }

  trace->save(options.tracePath);
  std::cout << "Saved trace: " << options.tracePath << "\n";

  for (int worker = 0; worker < options.workers; ++worker) {
    CHECK_HIP(hipFree(buffers[worker]));
    CHECK_HIP(hipStreamDestroy(streams[worker]));
  }
  return 0;
}
