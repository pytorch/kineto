# Profile multiple GPUs with TensorFlow 2.2:

## Software requirements

NVIDIA® `CUDA 10.2` must be installed on your system:

* [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) —`CUDA 10.2` requires `440.33 (Linux) / 441.22 (Windows)` and higher.
* [CUDA® Toolkit 10.2](https://developer.nvidia.com/cuda-toolkit-archive)
* CUPTI ships with the CUDA Toolkit.

## Linux setup

1. Install the [CUDA® Toolkit 10.2](https://developer.nvidia.com/cuda-downloads), select the target platform.
   Here's the an example to install cuda 10.2 on Ubuntu 16.04 with nvidia driver and cupti included.

   ```shell
   $ wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   $ sudo sh cuda_10.2.89_440.33.01_linux.run  # Select NVIDIA driver and CUPTI.
   ```

2. Ensure CUPTI exists on the path:
   ```shell
   $ /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcupti
   ```
   You should see a string like
   `libcupti.so.10.2 -> libcupti.so.10.2.75`

   If you don't have CUPTI on the path, prepend its installation directory to the $LD_LIBRARY_PATH environmental variable:

   ```shell
   $ export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
   ```
   Run the ldconfig command above again to verify that the `CUPTI 10.2` library is found.

3. Make symbolic link to `libcudart.so.10.1` and `libcupti.so.10.1`.
   TensorFlow 2.2 looks for those strings unless you build your own pip package with [TF_CUDA_VERSION=10.2](https://raw.githubusercontent.com/tensorflow/tensorflow/34bec1ebd4c7a2bc2cea5ea0491acf7615f8875e/tensorflow/tools/ci_build/release/ubuntu_16/gpu_py36_full/pip.sh).

   ```shell
   $ sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
   $ sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
   ```
4. Run the model again and look for `Successfully opened dynamic library libcupti.so.10.1` in the logs. Your setup is now complete.

## Alternative approach

If you cannot install `CUDA 10.2`, another option is via environment variable on GPU worker.

`$ export TF_GPU_CUPTI_USE_ACTIVITY_API=false`

However, this approach has higher measurement overhead (~20%). Therefore, this mode can be used for debugging and improving the
performance, but not recommend to compare and report absolute numbers.


## Known issues
* Multi-GPU Profiling does not work with `CUDA 10.1`. While `CUDA 10.2` is not officially supported by TF, profiling on `CUDA 10.2` is known to work on some configurations.
* Faking the symbolic links IS NOT a suggested way of using CUDA per NVIDIA's standard (the suggested way is to recompile TF with `CUDA 10.2` toolchain). But that gives a simple and easy way to try whether things work without spending a lot of time figuring out the compilation steps.
