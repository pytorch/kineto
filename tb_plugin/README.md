# PyTorch Profiler

### Quick Installation Instructions

* Clone the git repository

  `git clone https://github.com/pytorch/kineto.git`

* Navigate to the kineto/tb_plugin directory

* Install the profiler

  `pip install .`

* Verify installation is complete

  `pip list | grep torch-tb-profiler`

  Should display

  `torch-tb-profiler      0.1.0`


### Quick Start Instructions

* Start tensorboard

  Specify your profiling samples folder.
  Or you can specify kineto/tb_plugin/samples as an example.

  `tensorboard --logdir=./samples`

  If your web browser is not in the same machine that you start tensorboard,
  you can add `--bind_all` option, such as:

  `tensorboard --logdir=./samples --bind_all`

  Note: Make sure the default port 6006 is open to the browser's host.

* Open tensorboard in Chrome browser

  Open URL `http://localhost:6006` in the browser.

* Navigate to PYTORCH_PROFILER tab

  If the files under `--logdir` are too big or too many,
  please wait a while and refresh the browser to check latest loaded result.

### Quick Usage Instructions

We regard each running with profiler enabled as a "run".
In most cases a run is a single process. If DDP is enabled, then a run includes multiple processes.
We name each process a "worker".

Each run corresponds to a sub-folder under the folder specified by "--logdir".
Each sub-folder contains one or more chrome trace files, one for each process.
The kineto/tb_plugin/samples is an example of how the files are organized.

You can select the run and worker on the left control panel.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/control_panel.PNG)

Runs: Select a run. Each run is a PyTorch workload with profiling enabled.

Worker: Select a worker. Each worker is a process. There could be multiple workers when DPP is used.

Views: We organize the profiling result into multiple views,
from coarse-grained (overview-level) to fine-grained (kernel-level).

Currently we have the following performance diagnosis views:
- Overall View
- Operator View
- Kernel View
- Trace View

We describe each of these views below.

* Overall View

The overall view is a top level view of the process in your profiling run.
It shows an overview of time cost, including both host and GPU devices.
You can select the current worker in the left panel's "Workers" dropdown menu.

An example of overall view:
![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/overall_view.PNG)

Step Time Breakdown: This shows the performance summary. We regard each iteration (usually a mini-batch) as a step.
The time spent on each step is broken down into multiple categories as follows:

1. Kernel: Kernels execution time on GPU device;

2. Memcpy: GPU involved memory copy time (either D2D, D2H or H2D);

3. Memset: GPU involved memory set time;

4. Runtime: CUDA runtime execution time on host side;
Such as cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, ...

5. DataLoader: The data loading time spent in PyTorch DataLoader object;

6. CPU Exec: Host compute time, including every PyTorch operator running time;

7. Other: The time not included in any of the above.

Note: The summary of all the above categories is end-to-end wall-clock time.

The above list is ranked by priority from high to low. We count time in priority order.
The time cost with highest priority category(Kernel) is counted first,
then Memcpy, then Memset, ...,  and Other is counted last.
In the following example, the "Kernel" is counted first as 7-2=5 seconds;
Then the "Memcpy" is counted as 0 seconds, because it is fully hidden by "Kernel";
Then "CPU Exec" is counted as 2-1=1 seconds, because the [2,3] interval is hidden by "Kernel", only [1,2] interval is counted.

In this way, summarization of all the 7 categories' counted time in a step
will be the same with this step's total wall clock time.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/time_breakdown_priority.PNG)

Performance Recommendation: Leverage the profiling result to automatically highlight likely bottlenecks,
and give users actionable optimization suggestions.

* Operator View

This view displays the performance of every PyTorch operator that is executed either on the host or device.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/operator_view.PNG)
Each table row is a PyTorch operator, which is a computation operator implemented by C++,
such as “aten::relu_”, “aten::convolution”.

Calls: The operator's number of calls.

Device Self Duration: The accumulated time spent on GPU, not including this operator’s child operators.

Device Total Duration: The accumulated time spent on GPU, including this operator’s child operators.

Host Self Duration: The accumulated time spent on Host, not including this operator’s child operators.

Host Total Duration: The accumulated time spent on Host, including this operator’s child operators.

Note: Each above duration means wall-clock time. It doesn't mean the GPU or CPU during this period is fully utilized.

The top 4 pie charts are visualizations of the above 4 columns of durations.
They make the breakdowns visible at a glance.
Only the top N operators sorted by duration (configurable in the text box) will be shown in the pie charts.

The search box enables searching operators by name.

“Group By” could choose between “Operator” and “Operator + Input Shape”.
The “Input Shape” is shapes of tensors in this operator’s input argument list.
The empty “[]” means argument with scalar type.
For example, “[[32, 256, 14, 14], [1024, 256, 1, 1], [], [], [], [], [], [], []]”
means this operator has 9 input arguments,
1st is a tensor of size 32\*256\*14\*14,
2nd is a tensor of size 1024\*256\*1\*1,
the following 7 ones are scalar variables.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/operator_view_group_by_inputshape.PNG)

* Kernel View

This view shows all kernels’ time spent on GPU.
The time is calculated by subtracting the kernel's start time from the end time.

Note: This view does not include cudaMemcpy or cudaMemset. Because they are not kernels.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/kernel_view.PNG)

Total Duration: The accumulated time of all calls of this kernel.

Mean Duration: The average time duration of all calls. That's "Total Duration" divided by "Calls".

Max Duration: The maximum time duration among all calls.

Min Duration: The minimum time duration among all calls.

Note: This duration only includes a kernel's elapsed time on GPU device.
It does not mean the GPU is fully busy executing instructions during this time interval.
Some of the GPU cores may be idle due to reasons such as memory access latency or insufficient parallelism.
For example, there may be insufficient number of available warps per SM for the GPU to effectively
hide memory access latencies, or some SMs may be entirely idle due to an insufficient number of blocks.
Please refer to [Nvidia's best-practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

The top pie chart is a visualization of "Total Duration" column.
It makes the breakdowns visible at a glance.
Only the top N kernels sorted by accumulated time (configurable in the text box) will be shown in the pie chart.

The search box enables searching kernels by name.

“Group By” could choose between “Kernel Name” and “Kernel Name + Op Name”.
The "Operator" is the PyTorch operator which launches this kernel.

* Trace View

This view shows timeline using the chrome tracing plugin. Each horizontal area represents a thread or a CUDA stream.
Each colored rectangle represents an operator, or a CUDA runtime, or a GPU op which executes on GPU
(such as a kernel, a CUDA memory copy, a CUDA memory set, ...)

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/trace_view.PNG)

In the above example:

The “thread 0” is the CPU thread that do “backward” of neural network.

The “thread 1” is the main CPU thread, which mainly do data loading, forward of neural network, and model update.

The “stream 7” is a CUDA stream, which shows all kernels of this stream.

You can see there are 6 “ProfilerStep” at the top of "thread 1". Each “ProfilerStep” represents a mini-batch step.

The suspended toolbar has functionalities to help view the trace line.
For example, when the up-down arrow is enabled,
you can zoom in by dragging the mouse up and keeping mouse's left button pushed down.

![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/trace_view_one_step.PNG)

The “Optimizer.step#SGD.step” and ”enumerate(DataLoader)#_SingleProcessDataLoaderIter.\__next\__”
are high-level python side functions.

When you select the top-right corner's “Flow events” to ”async”,
you can see the relationship between an operator and its launched kernels.
![Alt text](https://github.com/pytorch/kineto/blob/master/tb_plugin/docs/images/trace_view_launch.PNG)
