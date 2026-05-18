#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "====: Working directory: $(pwd)"

# Ensure cmake is at least the max version needed by PyTorch and Kineto.
# Use conda-forge with strict priority so we get an up-to-date cmake.
# `--add channels` errors when the channel is already at the top of the
# user condarc, which happens on reused runners where a previous job
# already added it. Swallow that error so the script stays idempotent.
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority strict
conda config --show channels
conda install -y 'cmake>=3.27'
echo "====: Installed cmake version: $(cmake --version)"

python -m pip install --upgrade pip
echo "====: Installed pip version: $(python -m pip --version)"
