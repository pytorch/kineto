* GPU Utilization: GPU busy time / all steps time. The higher, the better. All steps time is the total time of all profiler steps(or called as iterations).
                   GPU busy time is the time during “all steps time” when is at least one GPU kernel running on this GPU.
                   However, this high-level utilization metric is coarse. It can’t tell how many SMs(Stream Multiprocessors) are in use.
                   For example, a kernel with a single thread running continuously will get 100% GPU utilization.

* Est. SM Efficiency: Estimated Stream Multiprocessor Efficiency. The higher, the better. This metric of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%).
                      This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, divided by “all steps time”.
                      It shows GPU Stream Multiprocessors’ utilization.
                      Although it is finer grained than above “GPU Utilization”, it still can’t tell the whole story.
                      For example, a kernel with only one thread per block can’t fully utilize each SM.

* Est. Achieved Occupancy: For most cases such as memory bandwidth bound kernels, a higher value often translates to better performance, especially when the initial value is very low. [Reference](http://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/S0514-GTC2012-GPU-Performance-Analysis.pdf). The definition of occupancy is [here](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm).
                           Occupancy is the ratio of active warps on an SM to the maximum number of
                           active warps supported by the SM. The theoretical occupancy of a kernel is upper limit occupancy of this kernel, limited by multiple
                           factors such as kernel shape, kernel used resource, and the GPU compute capability.
                           Est. Achieved Occupancy of a kernel, OCC_K = min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel).
                           This overall number is the weighted sum of all kernels OCC_K using kernel's execution duration as weight. It shows fine-grained low-level GPU utilization.

 * Kernel Time using Tensor Cores: Total GPU Time for Tensor Core kernels / Total GPU Time for all kernels. Higher is better.
                                   Tensor Cores are mixed precision floating point operations available for Volta GPUs (Titan V) and beyond.
                                   The cuDNN and cuBLAS libraries contain several Tensor Cores enabled GPU kernels for most Convolution and GEMM operations.
                                   This number shows Tensor Cores usage time ratio among all kernels on a GPU.
