#!/usr/bin/env python3
"""
Newlines linter - ensures files end with exactly one newline
"""

import argparse
import json
import sys
from typing import List


def check_newlines(filenames: List[str]) -> List[dict]:
    lint_messages = []

    for filename in filenames:
        try:
            with open(filename, "rb") as f:
                data = f.read()

            if len(data) == 0:
                continue

            if not data.endswith(b"\n"):
                lint_messages.append(
                    {
                        "path": filename,
                        "line": None,
                        "char": None,
                        "code": "NEWLINE",
                        "severity": "warning",
                        "name": "no-newline",
                        "original": data.decode("utf-8", errors="replace"),
                        "replacement": data.decode("utf-8", errors="replace") + "\n",
                        "description": "File does not end with a newline. Run `lintrunner -a` to fix.",
                    }
                )
            elif data.endswith(b"\n\n"):
                # File ends with multiple newlines
                stripped = data.rstrip(b"\n") + b"\n"
                lint_messages.append(
                    {
                        "path": filename,
                        "line": None,
                        "char": None,
                        "code": "NEWLINE",
                        "severity": "warning",
                        "name": "multiple-newlines",
                        "original": data.decode("utf-8", errors="replace"),
                        "replacement": stripped.decode("utf-8", errors="replace"),
                        "description": "File ends with multiple newlines. Run `lintrunner -a` to fix.",
                    }
                )
        except Exception as err:
            lint_messages.append(
                {
                    "path": filename,
                    "line": None,
                    "char": None,
                    "code": "NEWLINE",
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
        description="newlines linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    lint_messages = check_newlines(args.filenames)
    for lint_message in lint_messages:
        print(json.dumps(lint_message), flush=True)


if __name__ == "__main__":
    main()
