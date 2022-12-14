/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "nccl.h"

namespace kineto_stress_test {

inline void checkCudaStatus(cudaError_t status, int lineNumber = -1) {
  if (status != cudaSuccess) {
    printf(
        "PID %d --> CUDA API failed with status %d: %s at line %d\n",
        getpid(),
        status,
        cudaGetErrorString(status),
        lineNumber);
    exit(-1);
  }
}

#define MPICHECK(cmd) do {                              \
  int e = cmd;                                          \
  if( e != MPI_SUCCESS ) {                              \
    printf("PID %d --> Failed: MPI error %s:%d '%d'\n", \
        getpid(), __FILE__,__LINE__, e);                \
    exit(EXIT_FAILURE);                                 \
  }                                                     \
} while(0)

#define NCCLCHECK(cmd) do {                                 \
  ncclResult_t r = cmd;                                     \
  if (r!= ncclSuccess) {                                    \
    printf("PID %d --> Failed, NCCL error %s:%d '%s'\n",    \
        getpid(), __FILE__,__LINE__,ncclGetErrorString(r)); \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while(0)

struct tensor_pair {
  // Number of elements in the float arrays
  uint32_t n_elements;

  // If true, we pre-generate host buffers and use copy host to dev
  bool b_copy_h2d;

  // If true we download the output buffer to simulate output download
  bool b_copy_d2h;

  // GPU buffers
  float* d_A;
  float* d_B;
  float* d_C;

  // Host buffers
  float* h_A;
  float* h_B;
};

struct tensor_cache_args {
    // Sets GPU memory utilization
    uint32_t sz_cache_KB {1024 * 128};

    // If small, density is higher due to shorter kernel times
    uint32_t sz_min_tensor_KB {16};

    // If large, we will have kernels with high duration thus smaller
    // event density. That's because kernels will have to run on larger
    // buffer sizes.
    uint32_t sz_max_tensor_KB {2048};

    // Sets the maximum GPU memory
    uint32_t sz_GPU_memory_KB {1024 * 1024 * 16};

    // Simulates the chance of uploading a batch to the GPU.
    // It reduces event density if it's set too high
    double prob_h2d {0.005};

    // Simulates the chance of downloading results from the GPU.
    // It reduces event density if it's set too high
    double prob_d2h {0.0001};

    // Number of increments in the GPU memory usage to see what happens at the
    // peak memory usage.
    uint32_t num_increments {1};
    uint32_t num_pairs_per_increment {1};
};

// Generates all the buffer pairs, using a minimum and a maximum size.
// prob_h2d is the probability that we will generate a copy from host to
// device, in which case we need to pre-generate the buffers on the host.
void generate_tensor_cache(tensor_cache_args cache_args);

// For some experiments we may need to add additional pairs to stress the
// GPU memory limits
void add_pairs_to_tensor_cache(tensor_cache_args cache_args,
    uint32_t num_added_pairs);

// Size of the memory pool in kilobytes
extern uint32_t sz_memory_pool_KB;

// Number of tensor pairs in the memory pool
extern uint32_t num_tensor_pairs;

// The unique identifier used for NCCL communication
extern ncclUniqueId nccl_id;

// The communicator object used for communication
extern ncclComm_t nccl_communicator;

// NCCL buffers on the device
extern float* pBuffNCCLSend;
extern float* pBuffNCCLRecv;

// Re-initializes the random values in the device buffers
void re_initialize_buffer_values();

// Frees the host and device tensors
void free_tensor_cache();

struct stress_test_args {
  // Number of threads that run the stress test
    uint32_t num_workers {1};

    // Number of operations per stress test
    uint32_t num_operations {10000};

    // Can improve event density. Cuda streams per worker
    uint32_t num_cuda_streams {1};

    // Simulates cuda mallocs happening in the PT Cuda Cache Allocator
    double prob_cuda_malloc {0.001};

    // The min number of compute iterations in the cuda kernel. If high
    // this reduces event density.
    uint32_t min_iters_kernel {1};

    // The max number of compute iterations in the cuda kernel. If high
    // this reduces event density.
    uint32_t max_iters_kernel {5};

    // The probability that instead of a kernel call we do a memset on the
    // input buffers, using a magic value
    double memset_prob {0.05};

    // The min idle time between kernel launches in microseconds
    uint32_t min_idle_us {1};

    // The max idle time between kernel launches in microseconds
    uint32_t max_idle_us {2};

    // If true, we randomly sleep a number of microseconds between kernel
    // launches.
    bool simulate_host_time {false};

    // If non-zero, we allocate UVM memory and use it
    bool use_uvm_buffers {false};
    float* uvm_a {nullptr};
    float* uvm_b {nullptr};

    // Size of a single buffer in FP32 elements in UVM
    uint64_t uvm_len {0};

    // If set true, the UVM allocation and initialization will be done in parallel
    // with cache allocation (e.g. cudaHostAlloc)
    bool parallel_uvm_alloc {false};

    // The probability of running a kernel that uses UVM
    double uvm_kernel_prob {0.001};

    // If true we need to run the binary using MPI on multiple ranks
    bool is_multi_rank {false};

    // Number of parallel processes to be spawned via MPI
    int32_t num_ranks {1};

    // Use this variable to pin a process to a specific GPU index.
    // Do not modify!
    int32_t rank {0};

    // Size of the NCCL buffers which needs to be at least the size
    // of the largest tensor
    uint32_t sz_nccl_buff_KB {1024};

    // Number of iterations between NCCL sync calls
    uint32_t num_iters_nccl_sync {100};

    // If true, we pre-allocate CUDA streams and reuse them throughout
    // the experiment
    bool pre_alloc_streams {false};

    // The CUDA streams vector
    cudaStream_t *cuda_streams {nullptr};
};

void run_stress_test(
    uint32_t thread_id,
    uint32_t num_workers,
    stress_test_args test_args);

} // namespace kineto_stress_test
