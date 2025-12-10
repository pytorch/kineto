# Dynamic Test Exclusion System

This directory contains scripts for dynamically managing test exclusions in CI based on trunk test status.

## Overview

The `get_trunk_test_exclusions.py` script queries **PyTorch's trunk** test status to identify profiler tests that are currently failing in the main PyTorch repository. This prevents Kineto PRs from being blocked by pre-existing PyTorch profiler test failures while still maintaining test coverage for new issues introduced by the PR.

## How It Works

1. **Query PyTorch Test-Infra**: Fetches the disabled tests list from PyTorch's test-infra repository (`disabled-tests-condensed.json`)
2. **Filter Profiler Tests**: Extracts only profiler-related tests from the disabled list
3. **Check Recent Runs** (optional): Queries recent PyTorch trunk runs to catch very recent failures
4. **Generate Exclusions**: Creates `--deselect` arguments for pytest to exclude those tests
5. **Fallback List**: If the query fails or returns no results, it falls back to a hardcoded list of known problematic tests

## Usage

The script is automatically run by the `libkineto_cuda.yml` workflow in the "Generate dynamic test exclusions" step.

### Manual Testing

You can test the script locally:

```bash
cd /path/to/kineto
python3 .github/scripts/get_trunk_test_exclusions.py
```

This will output pytest `--deselect` arguments based on current trunk status.

### Requirements

- Python 3.6+
- `curl` command (for fetching PyTorch test-infra data)
- Network access to query PyTorch's test-infra repository

**Optional:**
- GitHub CLI (`gh`) for querying recent trunk runs (gracefully skipped if not available)

## Maintenance

### Updating the Fallback List

The fallback list in `get_trunk_test_exclusions.py` should be updated periodically based on PyTorch profiler test status:

1. Check current PyTorch profiler test status on their main branch
2. Update the `get_fallback_exclusions()` function with current known failures
3. As tests get fixed in PyTorch upstream, remove them from the fallback list

You can check PyTorch's test status at:
- PyTorch HUD: https://hud.pytorch.org
- Test-infra disabled tests: https://github.com/pytorch/test-infra/tree/generated-stats/stats

### Troubleshooting

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'gh'`
- **Cause**: GitHub CLI is not installed in the CI environment
- **Solution**: This is expected and handled gracefully - the script will skip GitHub API queries and use only the test-infra JSON
- **Impact**: Minimal - the test-infra JSON is the primary source and is updated every 15 minutes

**Issue**: Script returns empty exclusion list
- **Cause**: Network issues or PyTorch test-infra repository unavailable
- **Solution**: The script will automatically fall back to the hardcoded list
- **Check**: Verify network connectivity and that https://raw.githubusercontent.com/pytorch/test-infra/ is accessible

**Issue**: Tests are excluded that shouldn't be
- **Cause**: Recent main branch failures that are being investigated
- **Solution**: This is expected behavior - the system is working as designed
- **Alternative**: If you need to run a specific test, you can temporarily comment out the exclusion generation step in the workflow

## Benefits Over Hardcoded List

1. **Self-Updating**: Automatically adapts as trunk test status changes
2. **Less Maintenance**: No need to manually update exclusion lists
3. **Better Coverage**: As tests are fixed upstream, they're automatically re-enabled in PR testing
4. **Transparency**: Exclusion list is logged in each CI run for visibility

## Future Improvements

Potential enhancements:

1. **Issue-Based Tracking**: Adopt PyTorch's GitHub issue-based disabled test system
2. **Scheduled Updates**: Create a scheduled workflow to cache trunk status
3. **Test Stability Metrics**: Track how long tests have been failing
4. **Notifications**: Alert when tests are consistently failing on trunk
