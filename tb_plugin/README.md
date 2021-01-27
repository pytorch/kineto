# PyTorch Profiler

### Quick Installation Instructions

* Clone the git repository

  `git clone https://github.com/pytorch/kineto.git`

* Navigate to the plugin directory

* Install pytorch_profiler

  `pip install .`

* Verify installation is complete

  `pip list | grep tensorboard-plugin-torch-profiler`

  Should display

  `tensorboard-plugin-torch-profiler 0.1.0`


### Quick Start Instructions

* Start tensorboard

  Specify your profiling samples folder.
  Or you can specify <pytorch_profiler>/samples as an example.

  `tensorboard --logdir=./samples`

  If your web browser is not in the same machine that you start tensorboard,
  you can add `--bind_all` option, such as:

  `tensorboard --logdir=./samples --bind_all`

  Note: Make sure the default port 6006 is open to the browser's host.

* Open tensorboard in Chrome browser

  Open URL `http://localhost:6006` in the browser.

* Navigate to TORCH_PROFILER tab

  If the files under `--logdir` are too big or too many.
  Refresh the browser to check latest loaded result.

### Quick Usage Instructions

* Control Panel

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/control_panel.PNG)

Runs: Select a run. Each run is a PyTorch running with profiling.

Worker: Select a worker. Each worker is a process. There could be multiple workers under DDP running.

Views: We organize the profiling result into multiple view pages, from most coarse-grained (overview-level) to most fine-grained (kernel-level).

* Overall View

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/overall_view.PNG)

Step Time Breakdown: This shows high-level summary of performance. We regard each iteration (usually a mini-batch) as a step. Each step is broken into the multiple categories (with different colors) of where time is spent.
The main categories include:

1. Kernel: Kernels execution time on GPU device;

2. Memcpy: GPU involved memory copy time (either D2D, D2H or H2D);

3. Memset: GPU involved memory set time;

4. Runtime: CUDA runtime execution time on host side; Such as cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, ...

5. DataLoader: The data loading time spent in PyTorch DataLoader object;

6. CPU Exec: Host compute time, including every PyTorch operator running time;

7. Other: The time that is not included in any of the above.

Performance Recommendation: Leverage the profiling result to automatically get the bottlenecks and give suggestions to optimize. 
 
* Operator View

This view displays the performance of every Pytorch operator that is executed either on the host or device.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/operator_view.PNG)
Each table row is a “Pytorch operator”, which is a computation operator in C++ side, such as “aten::relu_”, “aten::convolution”.

Device Self Duration: The time spent on GPU(maybe not GPU fully utilized during this time range), not including this operator’s sub-functions.

Device Total Duration: The time spent on GPU(maybe not GPU fully utilized during this time range), including this operator’s sub-functions.

Host Self Duration: The time spent on Host(maybe not CPU busy during this time range), not including this operator’s sub-functions.

Host Total Duration: The time spent on Host(maybe not CPU busy during this time range), including this operator’s sub-functions.

The top 4 pie charts are visualizations of the above 4 columns. 

“Group By” could choose between “Operator” and “Operator + Input Shape”. The “Input Shape” is each operator’s input argument list’s tensor shapes, “[]” is scalar type. For example, “[[32, 256, 14, 14], [1024, 256, 1, 1], [], [], [], [], [], [], []]” means this operator has 9 input arguments, 1st is a tensor of size 32\*256\*14\*14, 2nd is a tensor of size 1024\*256\*1\*1, the following 7 ones are scalar variables.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/operator_view_group_by_inputshape.PNG)

* Kernel View

This view shows all kernels’ time spent on GPU. It is got from kernel’s end time subtract kernel’s start time. It does not include cudaMemcpy or cudaMemset now.

Note: This time just records a kernel's eplased time on GPU device. It does not mean GPU is fully busy during this time interval. That is, GPU occupancy may be less than 100% during this time interval. 

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/kernel_view.PNG)

Total Duration: The summarization time of all calls of this kernel.

Mean Duration: Total Duration/Calls.

Max Duration: Max{Duration of all Calls}.

Min Duration: Min{Duration of all Calls}.

The top pie is a visualization of "Total Duration" column. 

* Trace View

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view.PNG)

The “thread 0” is the CPU thread that do “backward” of neural network.

The “thread 1” is the main CPU thread, which mainly do data loading, forward of neural network, Optimizer.step.

The “stream 7” is CUDA stream trace line, you can see all kernels here.

We can see there are 6 “ProfilerStep” we captured.

We cat scroll the bar to zoom into it.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view_one_step.PNG)

The “Optimizer.step#SGD.step”、”enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__” are high-level python side operations.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view_launch.PNG)
When you select right-top corner's “Flow events” to ”async”, you can see which operator launched which GPU kernel.