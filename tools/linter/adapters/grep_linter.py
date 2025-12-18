#!/usr/bin/env python3
"""
Generic grep-based linter
"""

import argparse
import json
import re
import sys
from typing import List


def check_grep(
    filenames: List[str],
    pattern: str,
    linter_name: str,
    error_name: str,
    error_description: str,
) -> List[dict]:
    lint_messages = []

    regex = re.compile(pattern)

    for filename in filenames:
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    lint_messages.append(
                        {
                            "path": filename,
                            "line": line_num,
                            "char": None,
                            "code": linter_name,
                            "severity": "warning",
                            "name": error_name,
                            "original": line.rstrip("\n"),
                            "replacement": regex.sub("", line).rstrip("\n"),
                            "description": error_description,
                        }
                    )
        except Exception as err:
            lint_messages.append(
                {
                    "path": filename,
                    "line": None,
                    "char": None,
                    "code": linter_name,
                    "severity": "error",
                    "name": "command-failed",
                    "original": None,
                    "replacement": None,
                    "description": f"Failed to check {filename}: {err}",
                }
            )

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="grep-based linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="regex pattern to search for",
    )
    parser.add_argument(
        "--linter-name",
        required=True,
        help="name of the linter",
    )
    parser.add_argument(
        "--error-name",
        required=True,
        help="name of the error",
    )
    parser.add_argument(
        "--error-description",
        required=True,
        help="description of the error",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    lint_messages = check_grep(
        args.filenames,
        args.pattern,
        args.linter_name,
        args.error_name,
        args.error_description,
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message), flush=True)


if __name__ == "__main__":
    main()
