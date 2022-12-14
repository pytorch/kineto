/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <folly/dynamic.h>
#include <folly/json.h>

#include <libkineto.h>
#include <cuda.h>
#include <cupti_activity.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpi.h"
#include "kineto/libkineto/stress_test/random_ops_stress_test.cuh"

using namespace kineto_stress_test;

void read_inputs_from_json(std::string sJsonFile, stress_test_args *test_args,
    tensor_cache_args *cache_args) {
  std::ifstream fJson(sJsonFile.c_str());
  std::string sJson;

  if (!fJson.fail()) {
    fJson.seekg(0, std::ios::end);
    sJson.reserve(fJson.tellg());
    fJson.seekg(0, std::ios::beg);
    sJson.assign(
        (std::istreambuf_iterator<char>(fJson)),
        std::istreambuf_iterator<char>());
    fJson.close();

    folly::dynamic sJsonParsed = folly::parseJson(sJson);

    folly::dynamic jsonTestArgs = sJsonParsed["test_args"];
    test_args->num_operations = (uint32_t)jsonTestArgs["num_operations"].asInt();
    test_args->num_cuda_streams = (uint32_t)jsonTestArgs["num_cuda_streams"].asInt();
    test_args->prob_cuda_malloc = (double)jsonTestArgs["prob_cuda_malloc"].asDouble();
    test_args->min_iters_kernel = (uint32_t)jsonTestArgs["min_iters_kernel"].asInt();
    test_args->max_iters_kernel = (uint32_t)jsonTestArgs["max_iters_kernel"].asInt();
    test_args->memset_prob = (double)jsonTestArgs["memset_prob"].asDouble();
    test_args->min_idle_us = (uint32_t)jsonTestArgs["min_idle_us"].asInt();
    test_args->max_idle_us = (uint32_t)jsonTestArgs["max_idle_us"].asInt();
    test_args->simulate_host_time = (bool)jsonTestArgs["simulate_host_time"].asBool();
    test_args->num_workers = (uint32_t)jsonTestArgs["num_workers"].asInt();
    test_args->use_uvm_buffers = (bool)jsonTestArgs["use_uvm_buffers"].asBool();
    test_args->uvm_kernel_prob = (double)jsonTestArgs["uvm_kernel_prob"].asDouble();
    test_args->parallel_uvm_alloc = (bool)jsonTestArgs["parallel_uvm_alloc"].asBool();
    test_args->uvm_len = (uint64_t)jsonTestArgs["uvm_len"].asDouble();
    test_args->is_multi_rank = (bool)jsonTestArgs["is_multi_rank"].asBool();
    test_args->sz_nccl_buff_KB = (uint32_t)jsonTestArgs["sz_nccl_buff_KB"].asInt();
    test_args->num_iters_nccl_sync = (uint32_t)jsonTestArgs["num_iters_nccl_sync"].asInt();
    test_args->pre_alloc_streams = (bool)jsonTestArgs["pre_alloc_streams"].asBool();

    folly::dynamic cacheArgs = sJsonParsed["cache_args"];
    cache_args->sz_cache_KB = (uint32_t)cacheArgs["sz_cache_KB"].asInt();
    cache_args->sz_GPU_memory_KB = (uint32_t)cacheArgs["sz_GPU_memory_KB"].asInt();
    cache_args->sz_min_tensor_KB = (uint32_t)cacheArgs["sz_min_tensor_KB"].asInt();
    cache_args->sz_max_tensor_KB = (uint32_t)cacheArgs["sz_max_tensor_KB"].asInt();
    cache_args->prob_h2d = (double)cacheArgs["prob_h2d"].asDouble();
    cache_args->prob_d2h = (double)cacheArgs["prob_d2h"].asDouble();
    cache_args->num_increments = (uint32_t)cacheArgs["num_increments"].asInt();
    cache_args->num_pairs_per_increment = (uint32_t)cacheArgs["num_pairs_per_increment"].asInt();
  } else {
    printf("Reading input %s failed!.\n", sJsonFile.c_str());
  }
}

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
  std::string kTraceFile = "/tmp/kineto_stress_test_trace_";
  kTraceFile.append(std::to_string(getpid()));
  kTraceFile.append(".json");
  trace->save(kTraceFile);
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
    srand(time(nullptr));
    for (uint64_t i = 0; i < 32 * 128 * 1024; ++i) {
      uint64_t idx_a = rand() % test_args->uvm_len;
      uint64_t idx_b = rand() % test_args->uvm_len;
      test_args->uvm_a[idx_a] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      test_args->uvm_b[idx_b] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    std::cout << "UVM buffers initialized." << std::endl;
}

void run_parallel_stress_test(stress_test_args test_args) {
  std::vector<std::thread> v_workers;
  if (test_args.num_workers > 1) {
    v_workers.reserve(test_args.num_workers);
    for (int i = 0; i < test_args.num_workers; ++i) {
      v_workers.push_back(std::thread(run_stress_test, i, test_args.num_workers, test_args));
    }
    for (auto& t : v_workers) {
      t.join();
    }
    v_workers.clear();
  } else {
    run_stress_test(0, 1, test_args);
  }
}

int main(int argc, char *argv[]) {

  /////////////////////////////////////////////////////////////////////////////
  // Read test configuration
  /////////////////////////////////////////////////////////////////////////////

  int rank = 0;
  int num_ranks = 0;

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

  tensor_cache_args cache_args;
  stress_test_args test_args;

  // Parse input json file
  read_inputs_from_json(argv[1], &test_args, &cache_args);
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (test_args.is_multi_rank) {
    test_args.num_workers = 1;
    test_args.num_ranks = num_ranks;
    test_args.rank = rank;
    std::cout << "When running in multi-rank mode, only a single worker can be used!" << std::endl;
  } else {
    test_args.rank = 0;
    test_args.num_ranks = 1;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Start test
  /////////////////////////////////////////////////////////////////////////////

  // Initialize multi-processing
  if (test_args.is_multi_rank) {
    checkCudaStatus(cudaSetDevice(test_args.rank));

    // Broadcast ID to other ranks
    if (test_args.rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&nccl_id));
    }
    MPICHECK(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Create the communicator
    NCCLCHECK(ncclCommInitRank(&nccl_communicator, test_args.num_ranks, nccl_id, test_args.rank));

    // Allocate memory for the buffers
    checkCudaStatus(cudaMalloc(&pBuffNCCLSend, test_args.sz_nccl_buff_KB * 1024), __LINE__);
    checkCudaStatus(cudaMalloc(&pBuffNCCLRecv, test_args.sz_nccl_buff_KB * 1024), __LINE__);
    checkCudaStatus(cudaMemset(pBuffNCCLSend, 0, test_args.sz_nccl_buff_KB * 1024), __LINE__);
    checkCudaStatus(cudaMemset(pBuffNCCLRecv, 0, test_args.sz_nccl_buff_KB * 1024), __LINE__);

    // Make sure all the processes have allocated this memory
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Pre-allocate CUDA streams
  if (test_args.pre_alloc_streams) {
    test_args.cuda_streams = (cudaStream_t*)malloc(test_args.num_cuda_streams *
        test_args.num_workers * sizeof(cudaStream_t));
    for (uint32_t i = 0; i < test_args.num_cuda_streams * test_args.num_workers; ++i) {
      checkCudaStatus(cudaStreamCreateWithFlags(test_args.cuda_streams + i, cudaStreamNonBlocking), __LINE__);
    }
  }

  std::thread uvm_init_thread;
  if (test_args.use_uvm_buffers) {
    if (test_args.parallel_uvm_alloc) {
      uvm_init_thread = std::thread(uvm_allocation_thread, &test_args);
    } else {
      uvm_allocation_thread(&test_args);
      std::cout << "Rank " << test_args.rank << " finished UVM init." << std::endl;
    }
  }

  generate_tensor_cache(cache_args);
  std::cout << "Rank " << test_args.rank << " generating tensor cache completed." << std::endl;

  if (test_args.use_uvm_buffers) {
    if (test_args.parallel_uvm_alloc) {
      uvm_init_thread.join();
      std::cout << "Rank " << test_args.rank << " finished UVM init." << std::endl;
    }
  }

  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Warmup run
  uint32_t num_test_operations = test_args.num_operations;
  test_args.num_operations = 200;
  run_stress_test(0, 1, test_args);
  test_args.num_operations = num_test_operations;
  std::cout << "Rank " << test_args.rank << " warmup completed." << std::endl;

  // Re-generate tensor cache values
  re_initialize_buffer_values();
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Run without kineto tracing
  clock_t t_start = clock();
  run_parallel_stress_test(test_args);
  clock_t t_stop = clock();
  double t_no_trace = (double)(t_stop - t_start) / 1e+3;
  std::cout << "Rank " << test_args.rank << " no kineto test completed. Duration (ms) = "
      << t_no_trace << std::endl;

  // Re-generate tensor cache values
  re_initialize_buffer_values();
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Tracing thread
  std::thread kineto_thread;

  // We are gradually increasing the GPU memory usage so that we have GPU traces being
  // collected while we are almost out-of-memory. This is an attempt to expose errors
  // that we often see in our fleet like: illegal instruction, uncorrectable NVLink
  // error, etc.

  for (uint32_t idx = 0; idx < cache_args.num_increments; ++idx) {
    // Run with kineto tracing
    t_start = clock();
    kineto_thread = std::thread(trace_collection_thread);
    run_parallel_stress_test(test_args);
    kineto_thread.join();
    t_stop = clock();
    double t_with_trace = (double)(t_stop - t_start) / 1e+3;

    std::cout << "Rank " << test_args.rank << " kineto run " << idx
        << " completed. Used GPU memory (MB) = " << sz_memory_pool_KB / 1024
        << "; Duration (ms) = " << t_with_trace << std::endl;

    // The first run is the default run
    add_pairs_to_tensor_cache(cache_args, cache_args.num_pairs_per_increment);
  }

  // Re-generate tensor cache values
  re_initialize_buffer_values();
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Run again after kineto tracing
  t_start = clock();
  run_parallel_stress_test(test_args);
  t_stop = clock();
  double t_after_trace = (double)(t_stop - t_start) / 1e+3;
  std::cout << "Rank " << test_args.rank << " after kineto test completed. Duration (ms) = "
      << t_after_trace << std::endl;

  // Final barrier before destroying everything
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Destroy CUDA streams
  if (test_args.pre_alloc_streams) {
    for (int i = 0; i < test_args.num_cuda_streams * test_args.num_workers; ++i) {
      checkCudaStatus(cudaStreamSynchronize(test_args.cuda_streams[i]), __LINE__);
      checkCudaStatus(cudaStreamDestroy(test_args.cuda_streams[i]), __LINE__);
    }
    if (test_args.cuda_streams) {
      free(test_args.cuda_streams);
    }
  }

  // Free the tensor cache on the GPU
  free_tensor_cache();

  // Free UVM
  if (test_args.use_uvm_buffers) {
    checkCudaStatus(cudaFree(test_args.uvm_a), __LINE__);
    checkCudaStatus(cudaFree(test_args.uvm_b), __LINE__);
  }

  if (test_args.is_multi_rank) {
    checkCudaStatus(cudaFree(pBuffNCCLRecv));
    checkCudaStatus(cudaFree(pBuffNCCLSend));
    NCCLCHECK(ncclCommDestroy(nccl_communicator));
    MPICHECK(MPI_Finalize());
  }

  return 0;
}
