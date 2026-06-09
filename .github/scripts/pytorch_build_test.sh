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

# Enable sccache so PyTorch object files persist across CI runs. Most Kineto
# PRs touch only Kineto, so the bulk of PyTorch's thousands of objects are
# cache hits on warm runs. PyTorch's CMake picks up the compiler launchers
# below, so we only install the binary and point it at S3. Credentials come
# from the runner's AWS instance role, the same fleet PyTorch CI uses.
#
# All callers of this script run on Linux; select the binary by host
# architecture so the same script works on x86_64 and aarch64 runners. An
# unrecognized architecture warns and builds uncached rather than failing.
case "$(uname -m)" in
  x86_64)        SCCACHE_ARCH=x86_64 ;;
  aarch64|arm64) SCCACHE_ARCH=aarch64 ;;
  *) echo "====: Unsupported arch for sccache: $(uname -m); building uncached" >&2
     SCCACHE_ARCH="" ;;
esac

if [[ -n "${SCCACHE_ARCH}" ]]; then
  SCCACHE_VERSION="v0.8.2"
  SCCACHE_PKG="sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl"
  curl -fsSL "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_PKG}.tar.gz" | tar -xz -C /tmp
  install -m755 "/tmp/${SCCACHE_PKG}/sccache" /usr/local/bin/sccache

  export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
  export SCCACHE_REGION=us-east-1
  export SCCACHE_S3_KEY_PREFIX=kineto
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export CMAKE_C_COMPILER_LAUNCHER=sccache
  export CMAKE_CXX_COMPILER_LAUNCHER=sccache
  export CMAKE_CUDA_COMPILER_LAUNCHER=sccache
  sccache --start-server || true
  sccache --zero-stats || true
  echo "====: Enabled sccache (${SCCACHE_VERSION}, ${SCCACHE_ARCH})"
fi

# Build PyTorch from source
pip install -r requirements.txt

# Hipify PyTorch source code for ROCm build
if [[ "${GPU_ARCH}" == "rocm" ]]; then
  python tools/amd_build/build_amd.py
  echo "====: Hipified PyTorch source for ROCm"
fi

python -m pip install --no-build-isolation -v -e .
echo "====: Built PyTorch from source"

# Surface cache hit rates so warm-vs-cold runs are diagnosable from the log.
# Harmless when sccache was not enabled (no server running).
sccache --show-stats || true

# Download PyTorch's dynamic disabled tests list from S3. This is generated every
# 15 minutes from DISABLED GitHub Issues in pytorch/pytorch, enabling automatic
# skipping of known-broken/flaky tests without hardcoded deselections.
# The function downloads, processes (converts format and filters re-enabled issues),
# and caches the result to .pytorch-disabled-tests.json.
python -c "from tools.stats.import_test_stats import get_disabled_tests; get_disabled_tests('.')"
export DISABLED_TESTS_FILE=.pytorch-disabled-tests.json
echo "====: Downloaded disabled tests list"

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
