#!/usr/bin/env python3
"""
Script to dynamically generate pytest exclusion arguments based on PyTorch trunk test failures.

This script queries PyTorch's main repository and test-infra to identify profiler tests
that are currently failing on PyTorch trunk. It generates --deselect arguments to exclude
those tests from Kineto PR runs, preventing Kineto PRs from being blocked by pre-existing
PyTorch issues while maintaining test coverage for issues introduced by the Kineto changes.
"""

import json
import re
import subprocess
import sys
from typing import List, Set


def get_pytorch_profiler_test_status() -> Set[str]:
    """
    Query PyTorch's test infrastructure to get currently failing profiler tests.

    PyTorch maintains a list of disabled/unstable tests in their test-infra repo.
    We fetch this list to dynamically exclude tests that are known to be failing on trunk.
    """
    failed_tests = set()

    try:
        # Fetch PyTorch's disabled tests list from test-infra repo
        # This is updated every 15 minutes by PyTorch's infrastructure
        result = subprocess.run(
            [
                "curl",
                "-fsSL",
                "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/disabled-tests-condensed.json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        disabled_data = json.loads(result.stdout)

        # Look for profiler-related test entries
        # The JSON structure may have platform-specific entries
        for platform_key, tests in disabled_data.items():
            if isinstance(tests, list):
                for test_entry in tests:
                    if isinstance(test_entry, dict):
                        test_name = test_entry.get("name", "")
                    else:
                        test_name = str(test_entry)

                    # Filter for profiler tests
                    if "test/profiler/" in test_name or "profiler" in test_name.lower():
                        failed_tests.add(test_name)

        print(
            f"Found {len(failed_tests)} disabled profiler tests in PyTorch test-infra",
            file=sys.stderr,
        )

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not fetch PyTorch disabled tests: {e}", file=sys.stderr)

    return failed_tests


def get_recent_pytorch_trunk_runs(limit: int = 3) -> List[str]:
    """Fetch recent PyTorch trunk workflow run IDs that ran profiler tests."""
    try:
        # Query PyTorch's pull workflow which runs profiler tests
        result = subprocess.run(
            [
                "gh",
                "api",
                "repos/pytorch/pytorch/actions/workflows/pull.yml/runs",
                "--jq",
                f'.workflow_runs[] | select(.head_branch == "main" or .head_branch == "master") | .id | tostring',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        run_ids = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]
        return run_ids[:limit]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"Warning: GitHub CLI not available or failed: {e}", file=sys.stderr
        )
        print("Skipping recent trunk run checks (not critical)", file=sys.stderr)
        return []


def get_failed_tests_from_run(run_id: str) -> Set[str]:
    """Extract failed profiler test names from a PyTorch workflow run's logs."""
    failed_tests = set()

    try:
        # Get jobs for this PyTorch run
        # Look for jobs that run profiler tests (typically named with "test" in them)
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/pytorch/pytorch/actions/runs/{run_id}/jobs",
                "--jq",
                '.jobs[] | select(.name | contains("test")) | .id | tostring',
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        job_ids = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ][:5]  # Limit to first 5 jobs

        for job_id in job_ids:
            # Get job logs
            log_result = subprocess.run(
                ["gh", "api", f"repos/pytorch/pytorch/actions/jobs/{job_id}/logs"],
                capture_output=True,
                text=True,
                check=False,  # Logs might not be available for old runs
            )

            if log_result.returncode == 0:
                # Parse pytest output for FAILED profiler tests
                # Format: FAILED test/profiler/test_profiler.py::TestProfiler::test_kineto
                pattern = r"FAILED (test/profiler/[^\s]+)"
                matches = re.findall(pattern, log_result.stdout)
                failed_tests.update(matches)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"Warning: Failed to fetch logs for PyTorch run {run_id}: {e}",
            file=sys.stderr,
        )

    return failed_tests


def get_trunk_failures() -> Set[str]:
    """Get all profiler tests that are failing on PyTorch trunk."""
    print("Querying PyTorch test infrastructure...", file=sys.stderr)

    all_failures = set()

    # First, try to get disabled tests from PyTorch's test-infra (most reliable)
    disabled_tests = get_pytorch_profiler_test_status()
    if disabled_tests:
        all_failures.update(disabled_tests)

    # Optionally, also check recent trunk runs for additional failures
    # This catches very recent failures that might not be in disabled list yet
    run_ids = get_recent_pytorch_trunk_runs(limit=2)

    if run_ids:
        print(f"Checking {len(run_ids)} recent PyTorch trunk runs...", file=sys.stderr)
        for run_id in run_ids:
            failures = get_failed_tests_from_run(run_id)
            if failures:
                print(
                    f"PyTorch run {run_id}: {len(failures)} failed profiler tests",
                    file=sys.stderr,
                )
                all_failures.update(failures)

    return all_failures


def get_fallback_exclusions() -> Set[str]:
    """
    Fallback list of known problematic tests.
    Used when trunk query fails or returns no results.
    """
    return {
        "test/profiler/test_memory_profiler.py::TestDataFlow::test_data_flow_graph_complicated",
        "test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_sequential_fwd_bwd",
        "test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd",
        "test/profiler/test_memory_profiler.py::TestMemoryProfilerE2E::test_categories_e2e_simple_fwd_bwd_step",
        "test/profiler/test_profiler.py::TestProfiler::test_kineto",
        "test/profiler/test_profiler.py::TestProfiler::test_user_annotation",
        "test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize",
        "test/profiler/test_profiler.py::TestExperimentalUtils::test_profiler_debug_autotuner",
        "test/profiler/test_torch_tidy.py::TestTorchTidyProfiler::test_tensorimpl_invalidation_scalar_args",
    }


def generate_deselect_args(exclusions: Set[str]) -> str:
    """Generate pytest --deselect arguments from test names."""
    if not exclusions:
        return ""

    args = []
    for test in sorted(exclusions):  # Sort for consistent output
        args.append(f"--deselect={test}")

    # Return space-separated args (no line continuations - those break when used in variables)
    return " ".join(args)


def main():
    print("=" * 80, file=sys.stderr)
    print("Dynamic PyTorch Profiler Test Exclusion Script", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "This script queries PyTorch's trunk status to identify currently",
        file=sys.stderr,
    )
    print(
        "failing profiler tests, preventing Kineto PRs from being blocked",
        file=sys.stderr,
    )
    print("by pre-existing PyTorch issues.", file=sys.stderr)
    print("", file=sys.stderr)

    # Try to get PyTorch trunk failures
    trunk_failures = get_trunk_failures()

    if trunk_failures:
        print(
            f"\nFound {len(trunk_failures)} profiler tests failing on PyTorch trunk:",
            file=sys.stderr,
        )
        for test in sorted(trunk_failures):
            print(f"  - {test}", file=sys.stderr)
        exclusions = trunk_failures
    else:
        print("\nNo PyTorch trunk failures detected or query failed.", file=sys.stderr)
        print("Using fallback exclusion list.", file=sys.stderr)
        exclusions = get_fallback_exclusions()
        print(f"Fallback list contains {len(exclusions)} tests", file=sys.stderr)

    # Generate and output pytest arguments
    deselect_args = generate_deselect_args(exclusions)

    print("\n" + "=" * 80, file=sys.stderr)
    print("Generated pytest arguments:", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Output to stdout (this is what the workflow will use)
    print(deselect_args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
