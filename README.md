# Kineto

Kineto is a PyTorch performance profiling library (libkineto) focused on providing low-overhead full-system instrumentation for production workloads.
Libkineto is fully integrated with the PyToch Profiler, providing GPU profiling capabilities and in the future other system-level profiling. 
This repo also includes the PyTorch Profiler Tensorboard plugin, providing an easy-to-use end-to-end profiling experience.

## libkineto
Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the libkineto folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## PyTorch Tensorboard Profiler
The goal of the PyTorch Tensorboard plugin is to provide a seamless and intuitive end-to-end profiling experience, including straightforward collection from PyTorch and insightful visualizations and recommendations in the Tensorboard UI.
Please refer to the [README](tb_plugin/README.md) file in the tb_plugin folder.

## Future development:
- Support for tracing distributed workloads
- Better visibility into data loading and collectives
- Trace processing, analysis and recommendation engine
- System-level activities, multiple tracing sources
- Profiling and monitoring daemon for larger scale deployments 

##Releases and Contributing
We will follow the PyTorch release schedule which roughly happens on an every 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.

