# Kineto

Kineto is a library used in the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

The central component of Kineto is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## PyTorch TensorBoard Profiler (Deprecated)
> [!NOTE]
> The TensorBoard integration with PyTorch profiler (<code>tb_plugin</code> submodule) is deprecated and scheduled for permanent removal on 03/05/2026. 
> If you rely on <code>tb_plugin</code>, please comment on the <a href="https://github.com/pytorch/kineto/issues/1248">RFC issue</a> and consider migrating your workflow.  
> The code will be deleted after the feedback period.

The goal of the PyTorch TensorBoard Profiler is to provide a seamless and intuitive end-to-end profiling experience, including straightforward collection from PyTorch and insightful visualizations and recommendations in the TensorBoard UI.
Please refer to the [README](tb_plugin/README.md) file in the `tb_plugin` folder.

## Holistic Trace Analsysis
In order to compare Kineto traces across ranks, we reccomend using the [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) tool.

## Releases and Contributing
We will follow the PyTorch release schedule which roughly happens on a 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
