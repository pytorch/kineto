// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <libkineto.h>
#include <cuda.h>
#include <cupti_activity.h>
#include <sys/types.h>
#include <unistd.h>

#include "kineto/libkineto/sample_programs/random_ops_stress_test.cuh"

using namespace kineto_stress_test;

static const std::string kFileName = "/tmp/kineto_stress_test_trace.json";

void trace_collection_thread() {
  useconds_t t_trace_length_us = 2000000;

  // Configure CUPTI buffer sizes
  // size_t attrValue = 0, attrValueSize = sizeof(size_t);
  // attrValue = 3 * 1024 * 1024;
  // cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
  //    &attrValueSize, &attrValue);

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

void uvm_allocation_thread(stress_test_args *test_args) {
    uint64_t alloc_size = test_args->uvm_len * sizeof(float);

    std::cout << "UVM is used. Allocation size (MB) = " << 2 * alloc_size / (1024 * 1024) << std::endl;
    int currentDevice = 0;
    checkCudaStatus(cudaGetDevice(&currentDevice), __LINE__);
    checkCudaStatus(cudaMallocManaged((void**)&test_args->uvm_a, alloc_size, cudaMemAttachGlobal), __LINE__);
    checkCudaStatus(cudaMallocManaged((void**)&test_args->uvm_b, alloc_size, cudaMemAttachGlobal), __LINE__);
    checkCudaStatus(cudaMemAdvise((void*)test_args->uvm_a, alloc_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), __LINE__);
    checkCudaStatus(cudaMemAdvise((void*)test_args->uvm_b, alloc_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), __LINE__);
    checkCudaStatus(cudaMemAdvise((void*)test_args->uvm_a, alloc_size, cudaMemAdviseSetAccessedBy, currentDevice), __LINE__);
    checkCudaStatus(cudaMemAdvise((void*)test_args->uvm_b, alloc_size, cudaMemAdviseSetAccessedBy, currentDevice), __LINE__);
    std::cout << "UVM buffers allocated. Initializing them with values." << std::endl;

    // Put a bunch of non-zero values into the UVM buffers
    srand(time(NULL));
    for (uint64_t i = 0; i < 32 * 128 * 1024; ++i) {
      uint64_t idx_a = rand() % test_args->uvm_len;
      uint64_t idx_b = rand() % test_args->uvm_len;
      test_args->uvm_a[idx_a] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      test_args->uvm_b[idx_b] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    std::cout << "UVM buffers initialized." << std::endl;
}

int main() {
  tensor_cache_args cache_args;

  // Sets GPU memory utilization
  cache_args.sz_cache_KB = 16000000;

  // Sets the maximum GPU memory
  cache_args.sz_GPU_memory_KB = 16 * 1024 * 1024;

  // If small, density is higher due to shorter kernel times
  cache_args.sz_min_tensor_KB = 64;

  // If large, we will have kernels with high duration thus smaller
  // event density. That's because kernels will have to run on larger
  // buffer sizes.
  cache_args.sz_max_tensor_KB = 256;

  // Simulates the chance of uploading a batch to the GPU.
  // It reduces event density if it's set too high
  cache_args.prob_h2d = 0.0005;

  // Simulates the chance of downloading results from the GPU.
  // It reduces event density if it's set too high
  cache_args.prob_d2h = 0.0001;

  stress_test_args test_args;

  // Number of operations per stress test
  test_args.num_operations = 10000000;

  // Can improve event density
  test_args.num_cuda_streams = 10;

  // Simulates cuda mallocs happening in the PT Cuda Cache Allocator
  test_args.prob_cuda_malloc = 0.001;

  // The min number of compute iterations in the cuda kernel. If high
  // this reduces event density.
  test_args.min_iters_kernel = 1;

  // The max number of compute iterations in the cuda kernel. If high
  // this reduces event density.
  test_args.max_iters_kernel = 5;

  // The min idle time between kernel launches in microseconds
  test_args.min_idle_us = 0;

  // The max idle time between kernel launches in microseconds
  test_args.max_idle_us = 1;

  // If true, we randomly sleep a number of microseconds between kernel
  // launches.
  test_args.simulate_host_time = false;

  // If non-zero, we allocate UVM memory and use it
  test_args.use_uvm_buffers = false;

  // The probability of running a kernel that uses UVM
  test_args.uvm_kernel_prob = 0.05;

  // If set true, the UVM allocation and initialization will be done in parallel
  // with cache allocation (e.g. cudaHostAlloc)
  bool b_parallel_uvm_alloc = true;

  // Size of a single buffer in FP32 elements in UVM
  test_args.uvm_len = (uint64_t)(50 * 1024 * 1024) * (uint64_t)(1024 / 4);

  // Number of threads that run the stress test
  uint32_t num_workers = 1;

  // Number of increments in the GPU memory usage to see what happens at the
  // peak memory usage.
  uint32_t num_tensor_cache_increments = 1;
  uint32_t num_pairs_per_increment = 10;

  // Create more workers
  std::vector<std::thread> v_workers;
  if (test_args.use_uvm_buffers) {
    v_workers.reserve(num_workers + 1);
  } else {
    v_workers.reserve(num_workers);
  }

  // Allocate and initialize UVM in parallel with cudaHostAlloc
  if (test_args.use_uvm_buffers) {
    v_workers.push_back(std::thread(uvm_allocation_thread, &test_args));
  }

  if ((!b_parallel_uvm_alloc) && (test_args.use_uvm_buffers)) {
    // Wait for initialization to be finished
    for (auto& t : v_workers) {
      t.join();
    }
    v_workers.clear();
  }

  // Generate for non-kineto tests
  std::cout << "Generating tensor cache." << std::endl;
  generate_tensor_cache(cache_args);
  std::cout << "Finished generating tensor cache." << std::endl;

  // Wait for initialization to be finished
  if ((b_parallel_uvm_alloc) && (test_args.use_uvm_buffers)) {
    for (auto& t : v_workers) {
      t.join();
    }
    v_workers.clear();
  }

  // Warmup run
  std::cout << "Running warmup without kineto." <<  std::endl;
  uint32_t num_test_operations = test_args.num_operations;
  test_args.num_operations = 200;
  run_stress_test(0, 1, false, test_args);
  test_args.num_operations = num_test_operations;
  std::cout << "Warmup run finished." << std::endl;

  // Run without kineto tracing
  std::cout << "Running test without kineto. Num threads = " << num_workers << std::endl;
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
  std::cout << "Test without kineto completed." << std::endl;

  // Re-init the random values
  re_initialize_buffer_values();

  // Tracing thread
  std::thread kineto_thread;

  // We are gradually increasing the GPU memory usage so that we have GPU traces being
  // collected while we are almost out-of-memory. This is an attempt to expose errors
  // that we often see in our fleet like: illegal instruction, uncorrectable NVLink
  // error, etc.

  for (uint32_t idx = 0; idx < num_tensor_cache_increments; ++idx) {
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

    std::cout << "Run (" << idx << ") time with tracing enabled (ms): " <<
        t_with_trace << std::endl;
    std::cout << "Current usable memory pool size (MB) = " <<
        sz_memory_pool_KB / 1024 << std::endl;

    // The first run is the default run
    add_pairs_to_tensor_cache(cache_args, num_pairs_per_increment);
  }

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
  std::cout << "Execution time after tracing (ms): " << t_after_trace << std::endl;

  // Free the tensor cache on the GPU
  free_tensor_cache();

  // Free UVM
  if (test_args.use_uvm_buffers) {
    checkCudaStatus(cudaFree(test_args.uvm_a), __LINE__);
    checkCudaStatus(cudaFree(test_args.uvm_b), __LINE__);
  }

  return 0;
}
