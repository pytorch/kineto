#!/usr/bin/env bash
set -eux

GPU_ARCH="${1:?Usage: pytorch_build_test.sh <cpu|cuda|rocm>}"
SCRIPT_DIR="$(dirname "$0")"

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

# Build PyTorch from source
pip install -r requirements.txt
export USE_CUDA=1
export BUILD_TEST=1
python setup.py develop
echo "====: Built PyTorch from source"

# Load architecture-specific deselected tests
source "${SCRIPT_DIR}/deselected_tests_${GPU_ARCH}.sh"

DESELECT_ARGS=()
for t in "${DESELECTED_TESTS[@]}"; do
  DESELECT_ARGS+=(--deselect="$t")
done

# Run PyTorch profiler tests
pip install pytest
python -m pytest test/profiler/ -v "${DESELECT_ARGS[@]}"
popd
echo "====: Ran PyTorch profiler tests"
