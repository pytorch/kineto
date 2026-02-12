#!/usr/bin/env bash
set -eux

mkdir -p build_static build_shared

pushd build_static
cmake -DKINETO_LIBRARY_TYPE=static ../libkineto/
make -j
popd
echo "====: Compiled static libkineto"

pushd build_shared
cmake -DKINETO_LIBRARY_TYPE=shared ../libkineto/
make -j
popd
echo "====: Compiled shared libkineto"

pushd build_static
make test
popd
echo "====: Ran static libkineto tests"
