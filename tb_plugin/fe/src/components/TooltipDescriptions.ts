export const StepTimeBreakDownTooltip = `The time spent on each step is broken down into multiple categories as follows:
Kernel: Kernels execution time on GPU device;
Memcpy: GPU involved memory copy time (either D2D, D2H or H2D);
Memset: GPU involved memory set time;
Runtime: CUDA runtime execution time on host side; Such as cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, ...
DataLoader: The data loading time spent in PyTorch DataLoader object;
CPU Exec: Host compute time, including every PyTorch operator running time;
Other: The time not included in any of the above.`

export const DeviceSelfTimeTooltip = `The accumulated time spent on GPU, not including this operator’s child operators.`

export const DeviceTotalTimeTooltip = `The accumulated time spent on GPU, including this operator’s child operators.`

export const HostSelfTimeTooltip = `The accumulated time spent on Host, not including this operator’s child operators.`

export const HostTotalTimeTooltip = `The accumulated time spent on Host, including this operator’s child operators.`
