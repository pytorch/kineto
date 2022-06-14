// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <libkineto.h>
#include <cuda.h>
#include <sys/types.h>
#include <unistd.h>

#include "kineto/libkineto/sample_programs/random_ops_stress_test.cuh"

using namespace kineto_stress_test;

static const std::string kFileName = "/tmp/kineto_stress_test_trace.json";

void trace_collection_thread() {
  useconds_t t_trace_length_us = 2000000;

  // Config Kineto
  std::set<libkineto::ActivityType> types = {
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::OVERHEAD
  };
  auto& profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  profiler.prepareTrace(types);

  // Collect the trace
  profiler.startTrace();
  usleep(t_trace_length_us);
  auto trace = profiler.stopTrace();

  // Save the trace
  trace->save(kFileName);
}

int main() {
  // Test arguments
  stress_test_args test_args;
  test_args.num_operations = 10000000;
  test_args.num_cuda_streams = 10;
  test_args.prob_cuda_malloc = 0;
  test_args.min_iters_kernel = 1;
  test_args.max_iters_kernel = 5;
  test_args.min_idle_us = 0;
  test_args.max_idle_us = 1;
  test_args.simulate_host_time = false;

  // Worker count
  uint32_t num_workers = 1;

  // Create more workers
  std::vector<std::thread> v_workers;
  v_workers.reserve(num_workers);

  // Generate for non-kineto tests
  generate_tensor_cache();

  // Warmup run
  test_args.num_operations = 200;
  run_stress_test(0, 1, false, test_args);
  test_args.num_operations = 10000000;

  // Run without kineto tracing
  clock_t t_start = clock();
  for (int i = 0; i < num_workers; ++i) {
    v_workers.push_back(std::thread(run_stress_test, i, num_workers, false, test_args));
  }
  for (auto& t : v_workers) {
    t.join();
  }
  clock_t t_stop = clock();
  v_workers.clear();
  double t_no_trace = (double)(t_stop - t_start) / 1e+3;

  // Re-init the random values
  re_initialize_buffer_values();

  // Tracing thread
  std::thread kineto_thread;

  // Run with kineto tracing
  t_start = clock();
  for (int i = 0; i < num_workers; ++i) {
    if (i == 0) {
      // Will run kineto on a different thread
      kineto_thread = std::thread(trace_collection_thread);
    }

    // Run the iterations
    v_workers.push_back(std::thread(run_stress_test, i, num_workers, true, test_args));
  }
  for (auto& t : v_workers) {
    t.join();
  }
  kineto_thread.join();
  v_workers.clear();

  t_stop = clock();
  double t_with_trace = (double)(t_stop - t_start) / 1e+3;

  // Run again after kineto tracing
  t_start = clock();
  for (int i = 0; i < num_workers; ++i) {
    v_workers.push_back(std::thread(run_stress_test, i, num_workers, false, test_args));
  }
  for (auto& t : v_workers) {
    t.join();
  }
  t_stop = clock();
  v_workers.clear();
  double t_after_trace = (double)(t_stop - t_start) / 1e+3;

  std::cout << "Execution time before tracing (ms): " << t_no_trace << std::endl;
  std::cout << "Execution time with tracing enabled (ms): " << t_with_trace << std::endl;
  std::cout << "Execution time after tracing (ms): " << t_after_trace << std::endl;

  // Free the tensor cache on the GPU
  free_tensor_cache();

  return 0;
}
