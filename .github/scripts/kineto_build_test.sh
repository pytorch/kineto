#!/usr/bin/env bash
set -eux

GPU_ARCH="${1:?Usage: kineto_build_test.sh <cpu|cuda|rocm>}"
SCRIPT_DIR="$(dirname "$0")"

# Load architecture-specific cmake flags
source "${SCRIPT_DIR}/config_${GPU_ARCH}.sh"

mkdir -p build_static build_shared

pushd build_static
cmake "${KINETO_CMAKE_FLAGS[@]}" -DKINETO_LIBRARY_TYPE=static ../libkineto/
make -j
popd
echo "====: Compiled static libkineto"

pushd build_shared
cmake "${KINETO_CMAKE_FLAGS[@]}" -DKINETO_LIBRARY_TYPE=shared ../libkineto/
make -j
popd
echo "====: Compiled shared libkineto"

pushd build_static
make test
popd
echo "====: Ran static libkineto tests"
