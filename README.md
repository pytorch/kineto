# Kineto

> [!IMPORTANT]
> Development in this repository is frozen as of September 24, 2026 while Kineto is upstreamed into PyTorch. Please do not open new pull requests here. See [#1571](https://github.com/pytorch/kineto/issues/1571) for the migration timeline and where to submit future changes.

Kineto is a library used in the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

The central component of Kineto is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
