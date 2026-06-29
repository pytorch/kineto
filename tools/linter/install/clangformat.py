#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
clang-format installer for lintrunner
"""

import subprocess
import sys


def main() -> None:
    # Check if clang-format is already installed
    try:
        proc = subprocess.run(
            ["clang-format", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"clang-format is already installed: {proc.stdout.decode().strip()}")
        return
    except (OSError, subprocess.CalledProcessError):
        pass

    print("clang-format not found. Please install it:")
    print()
    print("On Ubuntu/Debian:")
    print("  sudo apt-get install clang-format")
    print()
    print("On macOS:")
    print("  brew install clang-format")
    print()
    print("Or install it from https://releases.llvm.org/")
    sys.exit(1)


if __name__ == "__main__":
    main()
