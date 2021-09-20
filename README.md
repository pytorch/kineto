# Kineto

Kineto is part of the PyTorch Profiler.

The Kineto project was started to help enable
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

A central component is libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

The PyTorch Profiler TensorBoard plugin provides powerful and intuitive visualizations of profiling results, as well as actionable recommendations, and is the best way to experience the new PyTorch Profiler.

## Libkineto
Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## PyTorch TensorBoard Profiler
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

