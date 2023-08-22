# Kineto

Kineto is part of the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

A central component is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## Holistic Trace Analysis

Holistic Trace Analysis (HTA) is an open source performance debugging library aimed at
distributed workloads. HTA takes as input PyTorch Profiler traces and elevates the performance
bottlenecks to enable faster debugging. Here's a partial list of features in HTA:

1. [Temporal Breakdown](https://hta.readthedocs.io/en/latest/source/features/temporal_breakdown.html): Breakdown of GPU time in terms of time spent in computation, communication, memory events, and idle time on a single node and across all ranks.
1. [Idle Time Breakdown](https://hta.readthedocs.io/en/latest/source/features/idle_time_breakdown.html): Breakdown of GPU idle time into waiting for the host, waiting for another kernel or attributed to an unknown cause.
1. [Kernel Breakdown](https://hta.readthedocs.io/en/latest/source/features/kernel_breakdown.html): Find kernels with the longest duration on each rank.
1. [Kernel Duration Distribution](https://hta.readthedocs.io/en/latest/source/features/kernel_breakdown.html#kernel-duration-distribution): Distribution of average time taken by longest kernels across different ranks.
1. [Communication Computation Overlap](https://hta.readthedocs.io/en/latest/source/features/comm_comp_overlap.html): Calculate the percentage of time when communication overlaps computation.

For a complete list see [here](http://hta.readthedocs.io).

## PyTorch TensorBoard Profiler (Deprecated)
The goal of the PyTorch TensorBoard Profiler is to provide a seamless and intuitive end-to-end profiling experience, including straightforward collection from PyTorch and insightful visualizations and recommendations in the TensorBoard UI.
Please refer to the [README](tb_plugin/README.md) file in the `tb_plugin` folder.

## Future Development Direction:
Some areas we're currently working on:
- Support for tracing distributed workloads
- Trace processing, analysis and recommendation engine
- System-level activities, multiple tracing sources
- Profiling and monitoring daemon for larger scale deployments

## Releases and Contributing
We will follow the PyTorch release schedule which roughly happens on a 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.

