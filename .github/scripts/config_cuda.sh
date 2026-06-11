#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Architecture-specific configuration for CUDA CI.
#
# This file is sourced by kineto_build_test.sh and pytorch_build_test.sh.
# It defines:
#   - Extra cmake flags for the libkineto build
#   - Environment variables for the PyTorch build
#   - Deselected pytest tests
#

# --- Kineto cmake flags ---

# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DKINETO_BACKEND=cuda
)

# --- PyTorch build environment variables ---

export USE_CUDA=1
export BUILD_TEST=1

# --- PyTorch build caching ---
# This arch's CI runner is an AWS instance, so it can reach PyTorch's shared
# S3 sccache bucket.
# shellcheck disable=SC2034
KINETO_USE_SCCACHE=1

# --- Deselected PyTorch profiler tests ---
# Each entry is a pytest node ID passed as a --deselect argument.
#
# Dynamic skipping of known-broken/flaky upstream tests is handled via
# DISABLED_TESTS_FILE in pytorch_build_test.sh. The hardcoded list below
# supplements it for tests not yet tracked upstream.

# shellcheck disable=SC2034
DESELECTED_TESTS=(
  test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize

  # https://github.com/pytorch/kineto/issues/1429
  test/profiler/test_profiler.py::TestProfilerDeviceCUDA
  test/profiler/test_profiler.py::TestProfilerDeviceCPU::test_forked_process_cpu

  # https://github.com/pytorch/kineto/issues/1430
  test/profiler/test_profiler.py::TestMetadataJsonFormat::test_kernel_metadata_field_types
  test/profiler/test_profiler.py::TestMetadataJsonFormat::test_kernel_metadata_has_expected_fields
  test/profiler/test_profiler.py::TestMetadataJsonFormat::test_metadata_json_is_valid_json_fragment
  test/profiler/test_profiler.py::TestMetadataJsonFormat::test_metadata_json_key_value_format
)
