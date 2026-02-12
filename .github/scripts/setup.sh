#!/usr/bin/env bash
set -eux

echo "====: Working directory: $(pwd)"

# Ensure cmake is at least the max version needed by PyTorch and Kineto
conda install -y cmake>=3.27
echo "====: Installed cmake version: $(cmake --version)"

python -m pip install --upgrade pip
echo "====: Installed pip version: $(python -m pip --version)"
