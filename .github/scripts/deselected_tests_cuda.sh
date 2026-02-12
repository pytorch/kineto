#!/usr/bin/env bash
# Deselected PyTorch profiler tests for CUDA CI.
#
# Each entry in the array is a pytest node ID that will be passed as a
# --deselect argument. Use standard bash comments to document why a test
# is excluded.
#
# TODO: Dynamically add/remove tests to the exclusion list based on their
# status on trunk instead of maintaining a hardcoded list of known failures.
# This will prevent the list from becoming stale as tests get fixed upstream.

DESELECTED_TESTS=(
  test/profiler/test_memory_profiler.py::TestDataFlow::test_data_flow_graph_complicated
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_sequential_fwd_bwd
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd
  test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd_step
  test/profiler/test_profiler.py::TestProfiler::test_kineto
  test/profiler/test_profiler.py::TestProfiler::test_user_annotation

  # https://github.com/pytorch/kineto/issues/1253
  test/profiler/test_profiler.py::TestProfiler::test_python_gc_event

  test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize
  test/profiler/test_profiler.py::TestExperimentalUtils::test_profiler_debug_autotuner
  test/profiler/test_torch_tidy.py::TestTorchTidyProfiler::test_tensorimpl_invalidation_scalar_args
)
