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
We name each process as a "worker".

Each run corresponds to a sub-folder under "--logdir" specified folder.
Under each run's sub-folder, each file is one worker's dumped chrome tracing file.
The kineto/tb_plugin/samples is an example of how to organize the files.

You can select run and worker on the left control panel.
   
![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/control_panel.PNG)

Runs: Select a run. Each run is a PyTorch running with profiling.

Worker: Select a worker. Each worker is a process. There could be multiple workers under DDP running.

Views: We organize the profiling result into multiple view pages, 
from most coarse-grained (overview-level) to most fine-grained (kernel-level).

Currently it has the following views to help user diagnose performance.
- Overall View
- Operator View
- Kernel View
- Trace View

The following will introduce these views.

* Overall View

The overall view is a top level view of the process in your profiling run. 
It shows an overview of time cost, including both host and GPU devices.
The process could be changed in left panel's "Workers" field.

An example of overall view:
![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/overall_view.PNG)

Step Time Breakdown: This shows performance summary. We regard each iteration (usually a mini-batch) as a step. 
Each step is broken into the multiple categories of time cost. The categories are as follows:

1. Kernel: Kernels execution time on GPU device;

2. Memcpy: GPU involved memory copy time (either D2D, D2H or H2D);

3. Memset: GPU involved memory set time;

4. Runtime: CUDA runtime execution time on host side; 
Such as cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, ...

5. DataLoader: The data loading time spent in PyTorch DataLoader object;

6. CPU Exec: Host compute time, including every PyTorch operator running time;

7. Other: The time that is not included in any of the above.

Note: The summary of all the above categories is end-to-end wall-clock time. 
We count time by priority. The time cost with highest priority category(Kernel) is counted firstly, 
then Memcpy, then Memset, ...,  and Other is last counted.
In the following example, the "Kernel" is counted firstly as 7-2=5 seconds; 
Then the "Memcpy" is counted as 0 seconds, because it is fully hidden by "Kernel"; 
Then "CPU Exec" is counted as 2-1=1 seconds, because the [2,3] interval is hidden by "Kernel", only [1,2] interval is counted.  
![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/time_breakdown_priority.PNG)

Performance Recommendation: Leverage the profiling result to automatically get the bottlenecks 
and give suggestions to optimize. 
 
* Operator View

This view displays the performance of every Pytorch operator that is executed either on the host or device.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/operator_view.PNG)
Each table row is a “Pytorch operator”, which is a computation operator in C++ side, 
such as “aten::relu_”, “aten::convolution”.

Calls: The operator's number of calls.

Device Self Duration: The accumulated time spent on GPU, not including this operator’s child operators.

Device Total Duration: The accumulated time spent on GPU, including this operator’s child operators.

Host Self Duration: The accumulated time spent on Host, not including this operator’s child operators.

Host Total Duration: The accumulated time spent on Host, including this operator’s child operators.

Note: Each above duration means wall-clock time. It doesn't mean the GPU or CPU during this period is fully utilized.

The top 4 pie charts are visualizations of the above 4 columns. 
They could show each operator's time percentage more straight forward.
You can change how many operators with top accumulated time to show in the pie charts. 

The search box enables searching operators by name.

“Group By” could choose between “Operator” and “Operator + Input Shape”. 
The “Input Shape” is each operator’s input argument list’s tensor shapes, 
“[]” is scalar type. For example, “[[32, 256, 14, 14], [1024, 256, 1, 1], [], [], [], [], [], [], []]” 
means this operator has 9 input arguments, 
1st is a tensor of size 32\*256\*14\*14, 
2nd is a tensor of size 1024\*256\*1\*1, 
the following 7 ones are scalar variables.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/operator_view_group_by_inputshape.PNG)

* Kernel View

This view shows all kernels’ time spent on GPU. The time is got from kernel’s end time minus kernel’s start time. 

Note: It does not include cudaMemcpy or cudaMemset now.

Note: This time just records a kernel's elapsed time on GPU device. 
It does not mean GPU is fully busy during this time interval. 
That is, GPU occupancy may be less than 100% during this time interval. 

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/kernel_view.PNG)

Total Duration: The accumulated time of all calls of this kernel.

Mean Duration: The average time duration among all calls. That's "Total Duration" divieded by "Calls".

Max Duration: The maximum time duration among all calls.

Min Duration: The minimum time duration among all calls.

The top pie is a visualization of "Total Duration" column. 
It could show each kernel's time percentage more straight forward.
You can change how many kernels with top accumulated time to show in the pie chart. 

The search box enables searching kernels by name.
“Group By” could choose between “Kernel Name” and “Kernel Name + Op Name”. 
The "Operator" is the operator which launches this kernel.

* Trace View

This view shows time line in chrome tracing. Each horizontal bar represents a thread or a CUDA stream.
Each range interval represents an operator, or a CUDA runtime, or a GPU op which executes on GPU 
such as a kernel, a CUDA memory copy, a CUDA memory set, ...

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view.PNG)

In the above example:

The “thread 0” is the CPU thread that do “backward” of neural network.

The “thread 1” is the main CPU thread, which mainly do data loading, forward of neural network, Optimizer.step.

The “stream 7” is CUDA stream trace line, you can see all kernels there.

You can see there are 6 “ProfilerStep” we captured.

When the up-down arrow is enabled, 
you can zoom in or zoom out by dragging the mouse up or down with mouse's left button pushed.

![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view_one_step.PNG)

The “Optimizer.step#SGD.step” and ”enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__”
are high-level python side operations.

When you select right-top corner's “Flow events” to ”async”, you can see which operator launched which GPU kernel.
![Alt text](https://github.com/pytorch/kineto/blob/tb_plugin/tb_plugin/docs/images/trace_view_launch.PNG)
