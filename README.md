# Kineto

Kineto is a PyTorch performance profiling library and framework focused on providing low-overhead full-system instrumentation for production workloads. For the moment, it consists of libkineto, an in-process profiling library integrated with PyTorch.
Going forward however there will be other related components added such as infrastructure for dameon-based deployment and trace processing pipelines.

## What is libkineto?
Libkineto, a component of the overall Kineto project, is an in-process profiling library which also provides a C++ API. Please refer to the [README](libkineto/README.md) file in the libkineto folder.

## Planned for initial release:
- libkineto, an in-process library providing CPU + GPU timeline tracing capabilities.
- An API allowing the PyTorch profiler to control timeline trace collection.
- Visualization in the Chrome browser using the chrome://tracing extension.

## Future development:
- Tensorboard integration 
- Collaboration features
- Daemon-based deployment for larger setups
- Distributed tracing support
- Trace processing and analysis pipeline
- System-level events, multiple tracing sources

## Releases and Contributing
We will follow the PyTorch release schedule which roughly happens on an every 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.

