#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

GPU_ARCH="${1:?Usage: pytorch_build_test.sh <cpu|cuda|rocm>}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Save kineto directory path before cloning PyTorch
KINETO_DIR=$(pwd)
echo "====: Kineto directory: ${KINETO_DIR}"

# Clone PyTorch and replace its Kineto with PR version
git clone --recursive --branch viable/strict https://github.com/pytorch/pytorch.git
echo "====: Cloned PyTorch"

pushd pytorch
rm -rf third_party/kineto
ln -s "${KINETO_DIR}" third_party/kineto
echo "====: Linked PR version of Kineto to PyTorch (${KINETO_DIR} -> third_party/kineto)"

# Load architecture-specific build env vars and deselected tests
# shellcheck source=/dev/null
source "${SCRIPTS_DIR}/config_${GPU_ARCH}.sh"

# Build PyTorch from source
pip install -r requirements.txt

# Hipify PyTorch source code for ROCm build
if [[ "${GPU_ARCH}" == "rocm" ]]; then
  python tools/amd_build/build_amd.py
  echo "====: Hipified PyTorch source for ROCm"
fi

python -m pip install --no-build-isolation -v -e .
echo "====: Built PyTorch from source"

# The deselected tests array is sourced from the architecture config above.
DESELECT_ARGS=()
for t in "${DESELECTED_TESTS[@]}"; do
  DESELECT_ARGS+=(--deselect="$t")
done

# Run PyTorch profiler tests
pip install pytest
python -m pytest test/profiler/ -v "${DESELECT_ARGS[@]}"
popd
echo "====: Ran PyTorch profiler tests"
