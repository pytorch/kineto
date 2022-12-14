/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <unistd.h>

#include "random_ops_stress_test.cuh"

namespace kineto_stress_test {

// Random number generation constants. They should not be modified, otherwise
// it's likely that most values will converge to 0.
#define LCG_A 8121
#define LCG_C 28411
#define LCG_M 134456
#define RNG_SEED_1 1025
#define RNG_SEED_2 2049

// We pre-create a memory pool of buffers on which we do various operations.
// This is similar to the tensor cache that PyTorch is managing.
tensor_pair* p_memory_pool;

// NCCL variables buffers
ncclUniqueId nccl_id;
ncclComm_t nccl_communicator;
float *pBuffNCCLSend;
float *pBuffNCCLRecv;

// Size of the memory pool in kilobytes
uint32_t sz_memory_pool_KB;

// Number of tensor pairs in the memory pool
uint32_t num_tensor_pairs;

// A kernel that fills a device buffer with random values
__global__ void simple_rng_lcg(float* d_A, int num_elements) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_elements) {
    uint32_t xn = tid * (tid + 1);
    d_A[tid] = (float)((LCG_A * xn + LCG_C) % LCG_M);
  }
}

// We are using this to reduce the number of code lines

struct lcg_kernel_input {
  float const* __restrict__ d_a;
  float const* __restrict__ d_b;
  float* __restrict__ d_c;
  int len;
  float const* __restrict__ uvm_a;
  float const* __restrict__ uvm_b;
  uint64_t uvm_len;
  int iters;
};

// C = A + B kernel where A and B are generated using a linear
// congruential generator. If the number of iterations is small
// the kernel is memory bandwidth bound. If iterations count is
// high, the kernel is compute bound.

// The kernel name is so long because we wanted to test if the number
// of characters in the kernel name influences the number of
// records that can be kept in the buffer.

// We use the template call to be able to change the kernel name with
// a simple hardcoded constant number

template<uint32_t offset_seed_a, uint32_t offset_seed_b>
__global__ void iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers(lcg_kernel_input input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < input.len) {
    uint32_t seed_a = (uint32_t)input.d_a[idx] + offset_seed_a;
    uint32_t seed_b = (uint32_t)input.d_b[idx] + offset_seed_b;
    uint32_t xna = 0;
    uint32_t xnb = 0;

    for (int i = 0; i < input.iters; ++i) {
      xna = (LCG_A * seed_a + LCG_C) % LCG_M;
      xnb = (LCG_A * seed_b + LCG_C) % LCG_M;
      seed_a = xna;
      seed_b = xnb;
    }

    input.d_c[idx] = 0.25 + (float)((xna + xnb) % 1000) / 1000.0;
  }
}

template<uint32_t offset_seed_a, uint32_t offset_seed_b>
__global__ void iterative_lcg_3_with_uvm(lcg_kernel_input input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t uvm_idx = idx;

  float val_uvm_a = 0.0f;
  float val_uvm_b = 0.0f;

  // Fetch data from UVM
  for (int i = 0; i < 5 ; ++i) {
    val_uvm_a += input.uvm_a[uvm_idx % input.uvm_len];
    val_uvm_b += input.uvm_b[uvm_idx % input.uvm_len];
    uvm_idx += 4096;
  }

  // Load device buffers, do some compute, save to device buffer
  if (idx < input.len) {
    uint32_t seed_a = (uint32_t)(input.d_a[idx] + val_uvm_a) + offset_seed_a;
    uint32_t seed_b = (uint32_t)(input.d_b[idx] + val_uvm_a) + offset_seed_b;
    uint32_t xna = 0;
    uint32_t xnb = 0;

    for (int i = 0; i < input.iters; ++i) {
      xna = (LCG_A * seed_a + LCG_C) % LCG_M;
      xnb = (LCG_A * seed_b + LCG_C) % LCG_M;
      seed_a = xna;
      seed_b = xnb;
    }

    input.d_c[idx] = 0.25 + (float)((xna + xnb) % 1000) / 1000.0;
  }
}

// Use this function to vary the kernel name at runtime
void call_compute_kernel(
  uint32_t thread_blocks,
  uint32_t threads_per_block,
  uint32_t shmem_sz,
  cudaStream_t stream,
  lcg_kernel_input kernel_args,
  uint32_t op_id
);

// Fill the buffers on the host with random values
void simple_lcg_host(float* h_A, int num_elements) {
  uint32_t xn = rand();
  for (int i = 0; i < num_elements; ++i) {
    h_A[i] = (float)((LCG_A * xn + LCG_C) % LCG_M);
    xn = h_A[i];
  }
}

void add_pairs_to_tensor_cache(tensor_cache_args cache_args, uint32_t
    num_added_pairs) {
  uint32_t num_current_pairs = num_tensor_pairs;

  for (uint32_t i = num_current_pairs;
      i < num_current_pairs + num_added_pairs; ++i) {
    uint32_t num_KB =
        rand() % (cache_args.sz_max_tensor_KB - cache_args.sz_min_tensor_KB) +
            cache_args.sz_min_tensor_KB;
    uint32_t num_elements = num_KB * 1024 / sizeof(float);

    // Allocate device buffers
    p_memory_pool[i].n_elements = num_elements;
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_A, num_elements * sizeof(float)), __LINE__);
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_B, num_elements * sizeof(float)), __LINE__);
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_C, num_elements * sizeof(float)), __LINE__);

    // Initialize device buffers with random values
    uint32_t thread_blocks = num_elements / 256;
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_A, p_memory_pool[i].n_elements);
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_B, p_memory_pool[i].n_elements);
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_C, p_memory_pool[i].n_elements);

    // Throw a dice to see if we will do memcopy host to device for this one and use pinned memory
    if (((float)(rand() % 100) / 100.0) < cache_args.prob_h2d) {
      p_memory_pool[i].b_copy_h2d = true;
      checkCudaStatus(cudaHostAlloc(&p_memory_pool[i].h_A, num_elements * sizeof(float), cudaHostAllocDefault), __LINE__);
      checkCudaStatus(cudaHostAlloc(&p_memory_pool[i].h_B, num_elements * sizeof(float), cudaHostAllocDefault), __LINE__);

      simple_lcg_host(p_memory_pool[i].h_A, num_elements);
      simple_lcg_host(p_memory_pool[i].h_B, num_elements);
    } else {
      p_memory_pool[i].b_copy_h2d = false;
      p_memory_pool[i].h_A = NULL;
      p_memory_pool[i].h_B = NULL;
    }

    // Simulate output download
    if (((float)(rand() % 10000) / 10000.0) < cache_args.prob_d2h) {
      p_memory_pool[i].b_copy_d2h = true;
    } else {
      p_memory_pool[i].b_copy_d2h = false;
    }

    // Now we have a new tensor pair
    num_tensor_pairs++;
    sz_memory_pool_KB += (3 * num_KB);
  }
}

void generate_tensor_cache(tensor_cache_args cache_args) {
  // Estimate the number of tensor pairs
  uint32_t num_pairs_max =
      cache_args.sz_GPU_memory_KB / (3 * (cache_args.sz_max_tensor_KB -
          cache_args.sz_min_tensor_KB) / 2);

  // Number of actual pairs
  num_tensor_pairs = 0;

  // At firs the pool is empty
  sz_memory_pool_KB = 0;

  // Pre-allocate num_pairs_max and if num_tensor_pairs comes lower, well,
  // that's life
  p_memory_pool =
      (tensor_pair*)malloc(num_pairs_max * sizeof(tensor_pair));

  // Start creating the pool
  srand(RNG_SEED_1);
  for (int i = 0; i < num_pairs_max; ++i) {
    // If we allocated too much, just exit
    if (sz_memory_pool_KB >= cache_args.sz_cache_KB) {
      printf("Allocated %d tensor pairs.\n", num_tensor_pairs);
      break;
    }

    add_pairs_to_tensor_cache(cache_args, 1);
  }
}

void re_initialize_buffer_values() {
  for (uint32_t i = 0; i < num_tensor_pairs; ++i) {
    uint32_t num_elements = p_memory_pool[i].n_elements;

    // Initialize device buffers with random values
    uint32_t thread_blocks = num_elements / 256;
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_A, p_memory_pool[i].n_elements);
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_B, p_memory_pool[i].n_elements);
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_C, p_memory_pool[i].n_elements);
  }
}

void free_and_realloc_tensor_pairs(tensor_pair *tensor_pair, cudaStream_t stream) {
// Older CUDA versions don't know about async malloc and free
#if defined(CUDA_VERSION) && CUDA_VERSION > 11000 && defined(ASYNC_MALLOC)

  checkCudaStatus(
    cudaFreeAsync(tensor_pair->d_A, stream),
        __LINE__);
  checkCudaStatus(
    cudaFreeAsync(tensor_pair->d_B, stream),
        __LINE__);
  checkCudaStatus(
    cudaFreeAsync(tensor_pair->d_C, stream),
        __LINE__);

  // Allocate device buffers
  uint32_t num_elements = tensor_pair->n_elements;
  checkCudaStatus(
    cudaMallocAsync(
        &tensor_pair->d_A,
        num_elements * sizeof(float),
        stream),
      __LINE__);
  checkCudaStatus(
    cudaMallocAsync(
        &tensor_pair->d_B,
        num_elements * sizeof(float),
        stream),
        __LINE__);
  checkCudaStatus(
    cudaMallocAsync(
        &tensor_pair->d_C,
        num_elements * sizeof(float),
        stream),
        __LINE__);

#else

  checkCudaStatus(cudaFree(tensor_pair->d_A), __LINE__);
  checkCudaStatus(cudaFree(tensor_pair->d_B), __LINE__);
  checkCudaStatus(cudaFree(tensor_pair->d_C), __LINE__);

  // Allocate device buffers
  uint32_t num_elements = tensor_pair->n_elements;
  checkCudaStatus(cudaMalloc(&tensor_pair->d_A,
    num_elements * sizeof(float)),
    __LINE__);
  checkCudaStatus(cudaMalloc(&tensor_pair->d_B,
    num_elements * sizeof(float)),
    __LINE__);
  checkCudaStatus(cudaMalloc(&tensor_pair->d_C,
    num_elements * sizeof(float)),
    __LINE__);

#endif // CUDA_VERSION >= 11000

  if (tensor_pair->b_copy_h2d) {
    checkCudaStatus(cudaFreeHost(tensor_pair->h_A), __LINE__);
    checkCudaStatus(cudaFreeHost(tensor_pair->h_B), __LINE__);

    checkCudaStatus(cudaHostAlloc(&tensor_pair->h_A, num_elements * sizeof(float), cudaHostAllocDefault), __LINE__);
    checkCudaStatus(cudaHostAlloc(&tensor_pair->h_B, num_elements * sizeof(float), cudaHostAllocDefault), __LINE__);

    simple_lcg_host(tensor_pair->h_A, num_elements);
    simple_lcg_host(tensor_pair->h_B, num_elements);
  }
}

void free_tensor_cache() {
  for (uint32_t i = 0; i < num_tensor_pairs; ++i) {
    checkCudaStatus(cudaFree(p_memory_pool[i].d_A), __LINE__);
    checkCudaStatus(cudaFree(p_memory_pool[i].d_B), __LINE__);
    checkCudaStatus(cudaFree(p_memory_pool[i].d_C), __LINE__);

    if (p_memory_pool[i].b_copy_h2d) {
      if (p_memory_pool[i].h_A) {
        checkCudaStatus(cudaFreeHost(p_memory_pool[i].h_A), __LINE__);
      }

      if (p_memory_pool[i].h_B) {
        checkCudaStatus(cudaFreeHost(p_memory_pool[i].h_B), __LINE__);
      }
    }
  }

  if (p_memory_pool) {
    free(p_memory_pool);
  }

  size_t mem_free = 0;
  size_t mem_total = 0;
  cudaMemGetInfo(&mem_free, &mem_total);
  size_t mem_used = (mem_total - mem_free) / 1024 / 1024;

  printf("GPU MB after freeing tensor cache: %6zu\n", mem_used);
}

void run_stress_test(
    uint32_t thread_id,
    uint32_t num_workers,
    stress_test_args test_args) {
  // We need to print an output to avoid making the compiler believe
  // that the following is a bunch of dead code.
  float checksum = 0.0;

  // Use a fixed random seed to be deterministic
  uint32_t rng_state = RNG_SEED_2 + thread_id;

  // Check memory usage
  size_t mem_free = 0;
  size_t mem_total = 0;
  size_t mem_used_before = 0;
  size_t mem_used_during = 0;
  checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
  mem_used_before = (mem_total - mem_free) / 1024 / 1024;

  // Create multiple streams
  cudaStream_t* v_streams = NULL;
  if (test_args.pre_alloc_streams) {
    v_streams = test_args.cuda_streams + (thread_id * test_args.num_cuda_streams);
  } else {
    v_streams = (cudaStream_t*)malloc(test_args.num_cuda_streams * sizeof(cudaStream_t));
    for (uint32_t i = 0; i < test_args.num_cuda_streams; ++i) {
      checkCudaStatus(cudaStreamCreate(v_streams + i), __LINE__);
    }
  }

  // Create output buffer for async downloads
  float* h_output = (float*)malloc(sizeof(float) * test_args.num_operations);
  memset(h_output, 0, test_args.num_operations * sizeof(float));

  // Measure time
  float t_wall_ms = 0.0;
  clock_t begin = clock();
  uint32_t num_pairs_per_worker = num_tensor_pairs / num_workers;

  // Start running the benchmark
  for (uint32_t i = 0; i < test_args.num_operations; ++i) {
    // All good things start with a break. In our case some GPU idle time
    if (test_args.simulate_host_time) {
      uint32_t gpu_idle_us = rand_r(&rng_state) % (test_args.max_idle_us -
          test_args.min_idle_us) + test_args.min_idle_us;
      usleep(gpu_idle_us);
    }

    // Generate stream ID and tensor pair index
    uint32_t local_pair_idx = i % num_pairs_per_worker;
    uint32_t pair_idx = thread_id * num_pairs_per_worker + local_pair_idx;
    uint32_t stream_idx = local_pair_idx % test_args.num_cuda_streams;

    // Check if we do a CUDA malloc
    if (((float)(rand_r(&rng_state) % 10000) / 10000.0) < test_args.prob_cuda_malloc) {
      checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
      free_and_realloc_tensor_pairs(p_memory_pool + pair_idx,
          v_streams[stream_idx]);

      // Initialize device buffers with random values
      uint32_t thread_blocks = p_memory_pool[pair_idx].n_elements / 256;
      simple_rng_lcg<<<thread_blocks, 256, 0, v_streams[stream_idx]>>>(
          p_memory_pool[pair_idx].d_A, p_memory_pool[pair_idx].n_elements);
      simple_rng_lcg<<<thread_blocks, 256, 0, v_streams[stream_idx]>>>(
          p_memory_pool[pair_idx].d_B, p_memory_pool[pair_idx].n_elements);
      simple_rng_lcg<<<thread_blocks, 256, 0, v_streams[stream_idx]>>>(
          p_memory_pool[pair_idx].d_C, p_memory_pool[pair_idx].n_elements);
    }

    // Do a CUDA memcpy if needed
    if (p_memory_pool[pair_idx].b_copy_h2d) {
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].d_A,
              p_memory_pool[pair_idx].h_A,
              p_memory_pool[pair_idx].n_elements * sizeof(float),
              cudaMemcpyHostToDevice,
              v_streams[stream_idx]),
          __LINE__);
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].d_B,
              p_memory_pool[pair_idx].h_B,
              p_memory_pool[pair_idx].n_elements * sizeof(float),
              cudaMemcpyHostToDevice,
              v_streams[stream_idx]),
          __LINE__);
    }

    // Launch kernel
    uint32_t num_iters_stream =
        rand_r(&rng_state) % (test_args.max_iters_kernel - test_args.min_iters_kernel) +
            test_args.min_iters_kernel;
    uint32_t thread_blocks = p_memory_pool[pair_idx].n_elements / 256;
    lcg_kernel_input kernel_args;
    kernel_args.d_a = p_memory_pool[pair_idx].d_A;
    kernel_args.d_b = p_memory_pool[pair_idx].d_B;
    kernel_args.d_c = p_memory_pool[pair_idx].d_C;
    kernel_args.len = p_memory_pool[pair_idx].n_elements;
    kernel_args.iters = num_iters_stream;

    if (test_args.use_uvm_buffers) {
      kernel_args.uvm_a = test_args.uvm_a;
      kernel_args.uvm_b = test_args.uvm_b;
      kernel_args.uvm_len = test_args.uvm_len;
    }
    else {
      kernel_args.uvm_a = NULL;
      kernel_args.uvm_b = NULL;
      kernel_args.uvm_len = 0;
    }

    bool b_do_memset = ((float)(rand_r(&rng_state) % 10000) / 10000.0) < test_args.memset_prob;
    bool b_uvm_kernel = ((float)(rand_r(&rng_state) % 10000) / 10000.0) < test_args.uvm_kernel_prob ? true : false;
    if ((kernel_args.uvm_len > 0) && (b_uvm_kernel)) {
      if (b_do_memset) {
        memset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float));
        memset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float));
        // checkCudaStatus(cudaMemset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float)));
        // checkCudaStatus(cudaMemset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float)));
      } else {
        iterative_lcg_3_with_uvm<113, 119><<<thread_blocks, 256, 0, v_streams[stream_idx]>>>(kernel_args);
      }
    } else {
      // Check to see if we do a simple kernel call or a memset
      if (b_do_memset) {
        checkCudaStatus(cudaMemset((void*)kernel_args.d_a, 42, kernel_args.len * sizeof(float)));
        checkCudaStatus(cudaMemset((void*)kernel_args.d_b, 42, kernel_args.len * sizeof(float)));
        checkCudaStatus(cudaMemset((void*)kernel_args.d_c, 42, kernel_args.len * sizeof(float)));
      } else {
        call_compute_kernel(thread_blocks, 256, 0, v_streams[stream_idx],
            kernel_args, i);
      }
    }

    // Simulate NCCL communication
    // printf("PID %d pair index %d.\n", getpid(), pair_idx);
    if ((i % test_args.num_iters_nccl_sync == 0) && (test_args.is_multi_rank)) {
      uint32_t n_elements = p_memory_pool[pair_idx].n_elements;
      size_t szTransfer = n_elements * sizeof(float);
      checkCudaStatus(cudaMemcpy(pBuffNCCLSend, p_memory_pool[pair_idx].d_C,
          szTransfer, cudaMemcpyDeviceToDevice), __LINE__);
      NCCLCHECK(ncclAllReduce((const void*)pBuffNCCLSend, (void*)pBuffNCCLRecv, n_elements,
          ncclFloat, ncclAvg, nccl_communicator, v_streams[stream_idx]));
      checkCudaStatus(cudaStreamSynchronize(v_streams[stream_idx]), __LINE__);
      checkCudaStatus(cudaMemcpy(p_memory_pool[pair_idx].d_C, pBuffNCCLRecv,
          szTransfer, cudaMemcpyDeviceToDevice), __LINE__);
    }

    // Simulate output download
    if (p_memory_pool[pair_idx].b_copy_d2h) {
      uint32_t rand_index = rand_r(&rng_state) % p_memory_pool[pair_idx].n_elements;
      checkCudaStatus(
          cudaMemcpyAsync(
              h_output + i,
              p_memory_pool[pair_idx].d_C + rand_index,
              sizeof(float),
              cudaMemcpyDeviceToHost,
              v_streams[stream_idx]),
          __LINE__);
    }

    // Get memory during execution
    if (i % 10000 == 0) {
      checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
      size_t mem_crnt = (mem_total - mem_free) / 1024 / 1024;
      if (mem_crnt >= mem_used_during) {
        mem_used_during = mem_crnt;
      }
    }
  }

  // Synchronize all streams
  for (int i = 0; i < test_args.num_cuda_streams; ++i) {
    checkCudaStatus(cudaStreamSynchronize(v_streams[i]), __LINE__);
  }

  // Measure execution time only until the streams are synchronized.
  // If we measure the time it takes to destroy them we may get high
  // run to run variation.

  clock_t end = clock();
  t_wall_ms = (double)(end - begin) / 1e+3;

  // Destroy the streams to avoid memory leaks
  if (!test_args.pre_alloc_streams) {
    for (int i = 0; i < test_args.num_cuda_streams; ++i) {
      checkCudaStatus(cudaStreamDestroy(v_streams[i]), __LINE__);
    }

    if (v_streams) {
      free(v_streams);
    }
  }

  // Compute a checksum to have some value as an output of the function
  for (int i = 0; i < test_args.num_operations; ++i) {
    checksum += h_output[i];
  }
  // checksum /= (float)test_args.num_operations;
  free(h_output);

  // Check how much memory we are using
  checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
  size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;

  printf(
      "Thread Index: %4d; GPU MB at Start: %6zu; Max GPU MB During Run: %6zu; GPU MB at Stop: %6zu; Runtime (ms): %6.3f; Checksum: %.5f\n",
      thread_id,
      mem_used_before,
      mem_used_during,
      mem_used_after,
      t_wall_ms,
      checksum);

  checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
}

// In case CUPTI compresses data using kernel name as a key to a hash map
// we want to see what happens in the case where we have lots of unique
// kernel names. This will make the trace to look like a rainbow.

void call_compute_kernel(
  uint32_t thread_blocks,
  uint32_t threads_per_block,
  uint32_t shmem_sz,
  cudaStream_t stream,
  lcg_kernel_input kernel_args,
  uint32_t op_id
) {
  switch (op_id % 20) {
    case 0:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<0, 1><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 1:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<1, 2><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 2:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<2, 3><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 3:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<3, 4><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 4:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<4, 5><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 5:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<5, 6><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 6:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<6, 7><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 7:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<7, 8><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 8:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<8, 9><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 9:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<9, 10><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 10:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<10, 11><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 11:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<11, 12><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 12:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<12, 13><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 13:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<13, 14><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 14:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<14, 15><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 15:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<15, 16><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 16:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<16, 17><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 17:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<17, 18><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 18:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<18, 19><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 19:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<19, 20><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 20:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<20, 1><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 21:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<21, 2><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 22:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<22, 3><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 23:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<23, 4><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 24:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<24, 5><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 25:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<25, 6><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 26:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<26, 7><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 27:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<27, 8><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 28:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<28, 9><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 29:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<29, 10><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 30:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<30, 11><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 31:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<31, 12><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 32:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<32, 13><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 33:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<33, 14><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 34:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<34, 15><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 35:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<35, 16><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 36:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<36, 17><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 37:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<37, 18><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 38:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<38, 19><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 39:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<39, 20><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    default:
      iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers_iterative_lcg_3_buffers<0, 0><<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
  }
}

} // namespace kineto_stress_test
