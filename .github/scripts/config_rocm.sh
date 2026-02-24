#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Architecture-specific configuration for ROCm CI.
#
# This file is sourced by kineto_build_test.sh and pytorch_build_test.sh.
# It defines:
#   - Extra cmake flags for the libkineto build
#   - Environment variables for the PyTorch build
#   - Deselected pytest tests
#

# --- Kineto cmake flags ---
# Enable ROCm (roctracer) and disable CUPTI. ROCM_SOURCE_DIR is required by
# CMakeLists.txt to locate roctracer headers and libraries.

# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DLIBKINETO_NOCUPTI=1
  -DROCM_SOURCE_DIR=/opt/rocm
)

# --- PyTorch build environment variables ---

export USE_ROCM=1
export BUILD_TEST=1

# --- Deselected PyTorch profiler tests ---
# Each entry is a pytest node ID passed as a --deselect argument.
#
# TODO: Dynamically add/remove tests to the exclusion list based on their
# status on trunk instead of maintaining a hardcoded list of known failures.
# This will prevent the list from becoming stale as tests get fixed upstream.

# shellcheck disable=SC2034
DESELECTED_TESTS=(
  test/profiler/test_memory_profiler.py::TestDataFlow::test_data_flow_graph_complicated
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_sequential_fwd_bwd
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd_step
  test/profiler/test_profiler.py::TestProfiler::test_kineto
  test/profiler/test_profiler.py::TestProfiler::test_user_annotation
  test/profiler/test_profiler.py::TestProfiler::test_python_gc_event
  test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize
  test/profiler/test_profiler.py::TestExperimentalUtils::test_profiler_debug_autotuner
  test/profiler/test_torch_tidy.py::TestTorchTidyProfiler::test_tensorimpl_invalidation_scalar_args

  # https://github.com/pytorch/kineto/issues/1242
  test/profiler/test_profiler.py::TestProfiler::test_disable_external_correlation

  # https://github.com/pytorch/kineto/issues/1243
  test/profiler/test_profiler.py::TestProfiler::test_profiler_cuda_sync_events

  # https://github.com/pytorch/kineto/issues/1241
  test/profiler/test_profiler.py::TestProfilerCUDA::test_mem_leak
)
