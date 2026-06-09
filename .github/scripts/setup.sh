#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "====: Working directory: $(pwd)"

python -m pip install --upgrade pip
echo "====: Installed pip version: $(python -m pip --version)"

# Ensure cmake is at least the max version needed by PyTorch and Kineto.
# Install from PyPI rather than conda: the cmake wheel ships a prebuilt
# binary and installs in seconds. The old conda path added the conda-forge
# channel with strict priority, which forced conda's classic solver to
# re-solve the entire base environment onto conda-forge and took over an
# hour on CI runners.
python -m pip install --upgrade 'cmake>=3.27'
echo "====: Installed cmake version: $(cmake --version)"
