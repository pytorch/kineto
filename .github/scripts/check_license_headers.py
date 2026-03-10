#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

REQUIRED_LICENSE_TEXT = "Copyright (c) Meta Platforms, Inc. and affiliates."

# Extensions that must use /* */ block comment style
C_STYLE_EXTENSIONS = {".cpp", ".h", ".cu", ".cuh"}

# Extensions that must use # comment style
HASH_STYLE_EXTENSIONS = {".py", ".sh"}

C_STYLE_HEADER = """\
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */"""

HASH_STYLE_HEADER = """\
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree."""


def check_license_header(file_path):
    """Check if a file has the required license header with the correct comment style."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, "error"

    if REQUIRED_LICENSE_TEXT not in content:
        return False, "missing"

    ext = os.path.splitext(file_path)[1]
    if ext in C_STYLE_EXTENSIONS:
        if "* " + REQUIRED_LICENSE_TEXT not in content:
            return False, "wrong_style"
    elif ext in HASH_STYLE_EXTENSIONS:
        if "# " + REQUIRED_LICENSE_TEXT not in content:
            return False, "wrong_style"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Check license headers in source files")
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args()

    if not args.files:
        return 0

    missing_headers = []
    wrong_style = []

    for file_path in args.files:
        ok, reason = check_license_header(file_path)
        if not ok:
            if reason == "wrong_style":
                wrong_style.append(file_path)
            else:
                missing_headers.append(file_path)

    has_errors = False

    if missing_headers:
        has_errors = True
        print("Missing license headers in the following files:")
        for file_path in missing_headers:
            print(f"  - {file_path}")

    if wrong_style:
        has_errors = True
        print("\nWrong comment style for license header in the following files:")
        for file_path in wrong_style:
            print(f"  - {file_path}")

    if has_errors:
        print("\nExpected license header for .cpp, .h, .cu, .cuh files:")
        print(C_STYLE_HEADER)
        print("\nExpected license header for .py, .sh files:")
        print(HASH_STYLE_HEADER)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
