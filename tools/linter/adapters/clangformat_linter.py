#!/usr/bin/env python3
"""
clang-format linter adapter for lintrunner
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_clangformat(
    files: List[str],
    binary: str,
) -> List[dict]:
    try:
        proc = subprocess.run(
            [binary, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.debug(f"clang-format version: {proc.stdout.decode().strip()}")
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            {
                "path": "<none>",
                "line": None,
                "char": None,
                "code": "CLANGFORMAT",
                "severity": "error",
                "name": "command-failed",
                "original": None,
                "replacement": None,
                "description": f"Failed to run {binary}: {err}",
            }
        ]

    lint_messages = []
    for filename in files:
        try:
            with open(filename, "rb") as f:
                original = f.read()

            proc = subprocess.run(
                [binary, filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            formatted = proc.stdout

            if original != formatted:
                lint_messages.append(
                    {
                        "path": filename,
                        "line": None,
                        "char": None,
                        "code": "CLANGFORMAT",
                        "severity": "warning",
                        "name": "format",
                        "original": original.decode("utf-8", errors="replace"),
                        "replacement": formatted.decode("utf-8", errors="replace"),
                        "description": "Run `lintrunner -a` to apply this patch.",
                    }
                )
        except Exception as err:
            lint_messages.append(
                {
                    "path": filename,
                    "line": None,
                    "char": None,
                    "code": "CLANGFORMAT",
                    "severity": "error",
                    "name": "command-failed",
                    "original": None,
                    "replacement": None,
                    "description": f"Failed to format {filename}: {err}",
                }
            )

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clang-format C++ linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-format binary path",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    lint_messages = run_clangformat(args.filenames, args.binary)
    for lint_message in lint_messages:
        print(json.dumps(lint_message), flush=True)


if __name__ == "__main__":
    main()
