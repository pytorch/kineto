# Libkineto

Libkineto is an in-process profiling library, part of the Kineto performance
tools project.

The library provides a way to collect GPU traces and metrics from the host
process, either via the library public API or by sending a signal, if enabled.

Currently only NVIDIA GPUs are supported.

## Examples

TODO

## Build Notes
Libkineto uses the standard CMAKE-based build flow.

### Dependencies
Libkineto requires gcc 5+.

+ ###### CUPTI
Libkineto uses CUPTI to collect traces and metrics from NVIDIA GPUs.

+ ###### fmt
fmt is used for its convenient and lightweight string formatting functionality.

+ ###### googletest
googletest is required to build and run Kineto's tests. **googletest is not
required** if you don't want to run Kineto tests. By default, building of tests
is **on**. Turn it off by setting KINETO\_BUILD\_TESTS to off.

You can download [CUPTI][1], [fmt][2], [googletest][3] and set
CUPTI\_SOURCE\_DIR, FMT\_SOURCE\_DIR, GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If the fmt and googletest variables are not set, cmake will
build the git submodules found in the third\_party directory.

General build instructions are as follows:

```
git clone --recursive https://github.com/pytorch/kineto.git
cd kineto/libkineto
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make
```

To run the tests after building libkineto (if tests are built), use the following
command:
```
make test
```

## Installing Libkineto
```
make install
```

## How Libkineto works
For a high-level overview, design philosophy and brief descriptions of various
parts of Libkineto please see [our blog][4]. [TODO]

## Full documentation
We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

## Join the Libkineto community
See the [`CONTRIBUTING`](CONTRIBUTING.md) file for how to help out.

## License
Libkineto is BSD licensed, as found in the [`LICENSE`](LICENSE) file.


[1]:https://developer.nvidia.com/CUPTI-CTK10_2
[2]:https://github.com/fmt
[3]:https://github.com/google/googletest
[4]:https://code.fb.com/ml-applications/kineto
