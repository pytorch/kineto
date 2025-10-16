/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string.h>

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#define NSEC_IN_SEC 1'000'000'000
#define A_VALUE 0.128f
#define B_VALUE 0.256f
#define MAX_EPS 1.0e-4f

static float Check(const std::vector<float>& a, float value) {
  assert(value > MAX_EPS);

  float eps = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    eps += std::fabs((a[i] - value) / value);
  }

  return eps / a.size();
}

// GEMM kernel function
void GEMM(
    const float* a,
    const float* b,
    float* c,
    unsigned size,
    sycl::id<2> id) {
  int i = id.get(0);
  int j = id.get(1);
  float sum = 0.0f;
  for (unsigned k = 0; k < size; ++k) {
    sum += a[i * size + k] * b[k * size + j];
  }
  c[i * size + j] = sum;
}

static float RunAndCheck(
    sycl::queue queue,
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    unsigned size,
    float expectedResult) {
  sycl::buffer<float, 1> a_buf(a.data(), a.size());
  sycl::buffer<float, 1> b_buf(b.data(), b.size());
  sycl::buffer<float, 1> c_buf(c.data(), c.size());

  [[maybe_unused]] sycl::event event = queue.submit([&](sycl::handler& cgh) {
    auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
    auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
    auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);

    // To be enabled
    if (true) {
      cgh.parallel_for<class __GEMM>(
          sycl::range<2>(size, size), [=](sycl::id<2> id) {
            auto a_acc_ptr = a_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto b_acc_ptr = b_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto c_acc_ptr = c_acc.get_multi_ptr<sycl::access::decorated::no>();
            GEMM(a_acc_ptr.get(), b_acc_ptr.get(), c_acc_ptr.get(), size, id);
          });
    }
  });
  queue.wait_and_throw();

  std::cout << "Matrix multiplication done. Checking result.." << std::endl;

  return Check(c, expectedResult);
}

void ComputeOnXpu(unsigned size, unsigned repeatCount) {
  unsigned sizeSq = size * size;
  sycl::device dev = sycl::device(sycl::gpu_selector_v);
  sycl::property_list propList{sycl::property::queue::in_order()};
  sycl::queue queue(
      dev, sycl::async_handler{}, propList); // Main runandcheck kernel

  std::cout << "DPC++ Matrix Multiplication (matrix size: " << size << " x "
            << size << ", repeats " << repeatCount << " times)" << std::endl;
  std::cout << "Target device: "
            << queue.get_info<sycl::info::queue::device>()
                   .get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<float> a(sizeSq, A_VALUE);
  std::vector<float> b(sizeSq, B_VALUE);
  std::vector<float> c(sizeSq, 0.0f);

  auto start = std::chrono::steady_clock::now();
  float expectedResult = A_VALUE * B_VALUE * size;

  for (unsigned i = 0; i < repeatCount; ++i) {
    float eps = RunAndCheck(queue, a, b, c, size, expectedResult);
    std::cout << "Results are " << ((eps < MAX_EPS) ? "" : "IN")
              << "CORRECT with accuracy: " << eps << std::endl;
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  std::cout << "Total execution time: " << time.count() << " sec" << std::endl;
}
