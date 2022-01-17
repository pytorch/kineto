/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export class MockAPI {
  runsGet() {
    return {
      runs: ['resnet50_num_workers_0', 'resnet50_num_workers_4'],
      loading: false
    }
  }

  viewsGet(run: string) {
    return Promise.resolve([
      'Overview',
      'Operator',
      'Kernel',
      'Trace',
      'Memory'
    ])
  }

  spansGet(run: string, view: String) {
    return Promise.resolve(['1', '2'])
  }

  workersGet(run: string, view: String) {
    return Promise.resolve(['worker0'])
  }

  overviewGet(run: string, worker: string, span: string) {
    return Promise.resolve({
      steps: {
        columns: [
          { type: 'string', name: 'Step' },
          { type: 'number', name: 'Kernel' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'Memcpy' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'Memset' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'Runtime' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'DataLoader' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'CPU Exec' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } },
          { type: 'number', name: 'Other' },
          { type: 'string', role: 'tooltip', p: { html: 'true' } }
        ],
        rows: [
          [
            '5',
            98598,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>Kernel: 98598us</b><br>Percentage: 52.46%</div>',
            1941,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>Memcpy: 1941us</b><br>Percentage: 1.03%</div>',
            90,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>Memset: 90us</b><br>Percentage: 0.05%</div>',
            2796,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>Runtime: 2796us</b><br>Percentage: 1.49%</div>',
            69317,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>DataLoader: 69317us</b><br>Percentage: 36.88%</div>',
            14091,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>CPU Exec: 14091us</b><br>Percentage: 7.5%</div>',
            1115,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 5<br>Total: 187948us<br><b>Other: 1115us</b><br>Percentage: 0.59%</div>'
          ],
          [
            '6',
            98570,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>Kernel: 98570us</b><br>Percentage: 56.28%</div>',
            1947,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>Memcpy: 1947us</b><br>Percentage: 1.11%</div>',
            89,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>Memset: 89us</b><br>Percentage: 0.05%</div>',
            2762,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>Runtime: 2762us</b><br>Percentage: 1.58%</div>',
            57669,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>DataLoader: 57669us</b><br>Percentage: 32.92%</div>',
            12968,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>CPU Exec: 12968us</b><br>Percentage: 7.4%</div>',
            1148,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 6<br>Total: 175153us<br><b>Other: 1148us</b><br>Percentage: 0.66%</div>'
          ],
          [
            '7',
            98596,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>Kernel: 98596us</b><br>Percentage: 54.86%</div>',
            1931,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>Memcpy: 1931us</b><br>Percentage: 1.07%</div>',
            91,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>Memset: 91us</b><br>Percentage: 0.05%</div>',
            2877,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>Runtime: 2877us</b><br>Percentage: 1.6%</div>',
            61257,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>DataLoader: 61257us</b><br>Percentage: 34.08%</div>',
            13768,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>CPU Exec: 13768us</b><br>Percentage: 7.66%</div>',
            1213,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 7<br>Total: 179733us<br><b>Other: 1213us</b><br>Percentage: 0.67%</div>'
          ],
          [
            '8',
            98623,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>Kernel: 98623us</b><br>Percentage: 56.5%</div>',
            1938,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>Memcpy: 1938us</b><br>Percentage: 1.11%</div>',
            89,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>Memset: 89us</b><br>Percentage: 0.05%</div>',
            2841,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>Runtime: 2841us</b><br>Percentage: 1.63%</div>',
            56453,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>DataLoader: 56453us</b><br>Percentage: 32.34%</div>',
            13420,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>CPU Exec: 13420us</b><br>Percentage: 7.69%</div>',
            1200,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 8<br>Total: 174564us<br><b>Other: 1200us</b><br>Percentage: 0.69%</div>'
          ],
          [
            '9',
            98504,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>Kernel: 98504us</b><br>Percentage: 54.07%</div>',
            1937,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>Memcpy: 1937us</b><br>Percentage: 1.06%</div>',
            87,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>Memset: 87us</b><br>Percentage: 0.05%</div>',
            2788,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>Runtime: 2788us</b><br>Percentage: 1.53%</div>',
            62690,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>DataLoader: 62690us</b><br>Percentage: 34.41%</div>',
            15025,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>CPU Exec: 15025us</b><br>Percentage: 8.25%</div>',
            1141,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 9<br>Total: 182172us<br><b>Other: 1141us</b><br>Percentage: 0.63%</div>'
          ],
          [
            '10',
            98641,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>Kernel: 98641us</b><br>Percentage: 59.43%</div>',
            1798,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>Memcpy: 1798us</b><br>Percentage: 1.08%</div>',
            88,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>Memset: 88us</b><br>Percentage: 0.05%</div>',
            3381,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>Runtime: 3381us</b><br>Percentage: 2.04%</div>',
            48185,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>DataLoader: 48185us</b><br>Percentage: 29.03%</div>',
            12773,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>CPU Exec: 12773us</b><br>Percentage: 7.7%</div>',
            1117,
            '<div class="visualization-tooltip" style="white-space: nowrap;">Step 10<br>Total: 165983us<br><b>Other: 1117us</b><br>Percentage: 0.67%</div>'
          ]
        ]
      },
      performance: [
        {
          name: 'Average Step Time',
          description: '',
          value: 177592,
          extra: 100,
          children: [
            { name: 'Kernel', description: '', value: 98589, extra: 55.51 },
            { name: 'Memcpy', description: '', value: 1915, extra: 1.08 },
            { name: 'Memset', description: '', value: 89, extra: 0.05 },
            { name: 'Runtime', description: '', value: 2908, extra: 1.64 },
            { name: 'DataLoader', description: '', value: 59262, extra: 33.37 },
            { name: 'CPU Exec', description: '', value: 13674, extra: 7.7 },
            { name: 'Other', description: '', value: 1156, extra: 0.65 }
          ]
        }
      ],
      recommendations:
        '<ul><li>This run has high time cost on input data loading. 33.4% of the step time is in DataLoader. You could try to set num_workers on DataLoader\'s construction and <a href ="https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading" target="_blank">enable multi-processes on data loading</a>.</li><li>Kernels with 68% time are launched by Tensor Cores eligible operators. You could enable <a href ="https://pytorch.org/docs/stable/amp.html" target="_blank">Automatic Mixed Precision</a> to speedup by using FP16.</li></ul>',
      environments: [
        { title: 'Number of Worker(s)', value: '1' },
        { title: 'Device Type', value: 'GPU' }
      ],
      gpu_metrics: {
        title: 'GPU Summary',
        data: [
          { title: 'GPU 0:', value: '' },
          { title: 'Name', value: 'Tesla V100-DGXS-32GB' },
          { title: 'Memory', value: '31.74 GB' },
          { title: 'Compute Capability', value: '7.0' },
          { title: 'GPU Utilization', value: '55.51 %' },
          { title: 'Est. SM Efficiency', value: '54.68 %' },
          { title: 'Est. Achieved Occupancy', value: '49.13 %' },
          { title: 'Kernel Time using Tensor Cores', value: '0.0 %' }
        ],
        tooltip:
          "The GPU usage metrics:\n\nGPU Utilization:\nGPU busy time / All steps time. The higher, the better. GPU busy time is the time during which there is at least one GPU kernel running on it. All steps time is the total time of all profiler steps(or called as iterations).\n\nEst. SM Efficiency:\nEstimated Stream Multiprocessor Efficiency. The higher, the better. This metric of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, divided by all steps time.\n\nEst. Achieved Occupancy:\nFor most cases such as memory bandwidth bounded kernels, the higher the better. Occupancy is the ratio of active warps on an SM to the maximum number of active warps supported by the SM. The theoretical occupancy of a kernel is upper limit occupancy of this kernel, limited by multiple factors such as kernel shape, kernel used resource, and the GPU compute capability.\nEst. Achieved Occupancy of a kernel, OCC_K = min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). This overall number is the weighted average of all kernels' OCC_K using kernel's execution duration as weight. It shows fine-grained low-level GPU utilization.\n\nKernel using Tensor Cores:\nTotal GPU Time for Tensor Core kernels / Total GPU Time for all kernels.\n"
      }
    })
  }

  diffnodeGet(
    run: string,
    worker: string,
    span: string,
    exp_run: string,
    exp_worker: string,
    exp_span: string,
    path?: string
  ) {
    return Promise.resolve({
      left: {
        name: 'multiple nodes',
        duration: 4246748,
        device_duration: 376761,
        total_duration: 3823182,
        aggs: [
          {
            name: 'aten::empty',
            calls: 4214,
            host_duration: 186312,
            device_duration: 0,
            self_host_duration: 186312,
            self_device_duration: 0
          },
          {
            name: 'aten::zero_',
            calls: 846,
            host_duration: 31902,
            device_duration: 736,
            self_host_duration: 17460,
            self_device_duration: 0
          },
          {
            name: 'aten::zeros',
            calls: 520,
            host_duration: 62713,
            device_duration: 0,
            self_host_duration: 32640,
            self_device_duration: 0
          },
          {
            name: 'aten::to',
            calls: 2696,
            host_duration: 1711486,
            device_duration: 8796,
            self_host_duration: 37162,
            self_device_duration: 0
          },
          {
            name: 'detach',
            calls: 256,
            host_duration: 4379,
            device_duration: 0,
            self_host_duration: 4379,
            self_device_duration: 0
          },
          {
            name: 'aten::detach',
            calls: 256,
            host_duration: 10596,
            device_duration: 0,
            self_host_duration: 6217,
            self_device_duration: 0
          },
          {
            name: 'aten::as_strided',
            calls: 914,
            host_duration: 8470,
            device_duration: 0,
            self_host_duration: 8470,
            self_device_duration: 0
          },
          {
            name: 'aten::unsqueeze',
            calls: 384,
            host_duration: 19150,
            device_duration: 0,
            self_host_duration: 16142,
            self_device_duration: 0
          },
          {
            name: 'aten::empty_strided',
            calls: 1158,
            host_duration: 50043,
            device_duration: 0,
            self_host_duration: 50043,
            self_device_duration: 0
          },
          {
            name: 'aten::copy_',
            calls: 1412,
            host_duration: 1518205,
            device_duration: 8796,
            self_host_duration: 1509009,
            self_device_duration: 8796
          },
          {
            name: 'aten::_to_copy',
            calls: 1284,
            host_duration: 1674324,
            device_duration: 8796,
            self_host_duration: 104788,
            self_device_duration: 0
          },
          {
            name: 'aten::upsample_bilinear2d',
            calls: 128,
            host_duration: 460479,
            device_duration: 0,
            self_host_duration: 421547,
            self_device_duration: 0
          },
          {
            name: 'aten::squeeze',
            calls: 128,
            host_duration: 9401,
            device_duration: 0,
            self_host_duration: 8211,
            self_device_duration: 0
          },
          {
            name: 'aten::round',
            calls: 128,
            host_duration: 31311,
            device_duration: 0,
            self_host_duration: 31311,
            self_device_duration: 0
          },
          {
            name: 'aten::slice',
            calls: 260,
            host_duration: 17762,
            device_duration: 0,
            self_host_duration: 15082,
            self_device_duration: 0
          },
          {
            name: 'detach_',
            calls: 512,
            host_duration: 4194,
            device_duration: 0,
            self_host_duration: 4194,
            self_device_duration: 0
          },
          {
            name: 'aten::detach_',
            calls: 512,
            host_duration: 14514,
            device_duration: 0,
            self_host_duration: 10320,
            self_device_duration: 0
          },
          {
            name: 'aten::result_type',
            calls: 640,
            host_duration: 1734,
            device_duration: 0,
            self_host_duration: 1734,
            self_device_duration: 0
          },
          {
            name: 'aten::pow',
            calls: 640,
            host_duration: 86249,
            device_duration: 0,
            self_host_duration: 78373,
            self_device_duration: 0
          },
          {
            name: 'aten::sub',
            calls: 640,
            host_duration: 183533,
            device_duration: 0,
            self_host_duration: 75637,
            self_device_duration: 0
          },
          {
            name: 'aten::gt',
            calls: 640,
            host_duration: 71284,
            device_duration: 0,
            self_host_duration: 49575,
            self_device_duration: 0
          },
          {
            name: 'aten::_local_scalar_dense',
            calls: 768,
            host_duration: 4948,
            device_duration: 0,
            self_host_duration: 4948,
            self_device_duration: 0
          },
          {
            name: 'aten::item',
            calls: 768,
            host_duration: 20922,
            device_duration: 0,
            self_host_duration: 15974,
            self_device_duration: 0
          },
          {
            name: 'aten::is_nonzero',
            calls: 640,
            host_duration: 27934,
            device_duration: 0,
            self_host_duration: 10747,
            self_device_duration: 0
          },
          {
            name: 'aten::div',
            calls: 130,
            host_duration: 168214,
            device_duration: 75,
            self_host_duration: 146203,
            self_device_duration: 75
          },
          {
            name: 'aten::resize_',
            calls: 6,
            host_duration: 248,
            device_duration: 0,
            self_host_duration: 248,
            self_device_duration: 0
          },
          {
            name: 'aten::narrow',
            calls: 4,
            host_duration: 280,
            device_duration: 0,
            self_host_duration: 99,
            self_device_duration: 0
          },
          {
            name: 'aten::_cat',
            calls: 4,
            host_duration: 92993,
            device_duration: 0,
            self_host_duration: 92405,
            self_device_duration: 0
          },
          {
            name: 'aten::cat',
            calls: 4,
            host_duration: 93282,
            device_duration: 0,
            self_host_duration: 289,
            self_device_duration: 0
          },
          {
            name: 'aten::stack',
            calls: 4,
            host_duration: 124757,
            device_duration: 0,
            self_host_duration: 22050,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_convolution',
            calls: 106,
            host_duration: 44043,
            device_duration: 71832,
            self_host_duration: 35027,
            self_device_duration: 71832
          },
          {
            name: 'aten::_convolution',
            calls: 106,
            host_duration: 51312,
            device_duration: 71832,
            self_host_duration: 7269,
            self_device_duration: 0
          },
          {
            name: 'aten::convolution',
            calls: 106,
            host_duration: 55287,
            device_duration: 71832,
            self_host_duration: 3975,
            self_device_duration: 0
          },
          {
            name: 'aten::conv2d',
            calls: 106,
            host_duration: 59323,
            device_duration: 71832,
            self_host_duration: 4036,
            self_device_duration: 0
          },
          {
            name: 'aten::add',
            calls: 138,
            host_duration: 17461,
            device_duration: 10540,
            self_host_duration: 15188,
            self_device_duration: 10540
          },
          {
            name: 'aten::empty_like',
            calls: 108,
            host_duration: 11504,
            device_duration: 0,
            self_host_duration: 4865,
            self_device_duration: 0
          },
          {
            name: 'aten::view',
            calls: 214,
            host_duration: 3589,
            device_duration: 0,
            self_host_duration: 3589,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_batch_norm',
            calls: 106,
            host_duration: 71328,
            device_duration: 25802,
            self_host_duration: 40944,
            self_device_duration: 25802
          },
          {
            name: 'aten::_batch_norm_impl_index',
            calls: 106,
            host_duration: 76354,
            device_duration: 25802,
            self_host_duration: 5026,
            self_device_duration: 0
          },
          {
            name: 'aten::batch_norm',
            calls: 106,
            host_duration: 79832,
            device_duration: 25802,
            self_host_duration: 3478,
            self_device_duration: 0
          },
          {
            name: 'aten::clamp_min',
            calls: 98,
            host_duration: 5417,
            device_duration: 12000,
            self_host_duration: 3885,
            self_device_duration: 12000
          },
          {
            name: 'aten::clamp_min_',
            calls: 98,
            host_duration: 8537,
            device_duration: 12000,
            self_host_duration: 3120,
            self_device_duration: 0
          },
          {
            name: 'aten::relu_',
            calls: 98,
            host_duration: 16708,
            device_duration: 12000,
            self_host_duration: 8171,
            self_device_duration: 0
          },
          {
            name: 'aten::max_pool2d_with_indices',
            calls: 2,
            host_duration: 442,
            device_duration: 940,
            self_host_duration: 405,
            self_device_duration: 940
          },
          {
            name: 'aten::max_pool2d',
            calls: 2,
            host_duration: 542,
            device_duration: 940,
            self_host_duration: 100,
            self_device_duration: 0
          },
          {
            name: 'aten::add_',
            calls: 998,
            host_duration: 72931,
            device_duration: 13090,
            self_host_duration: 57558,
            self_device_duration: 13090
          },
          {
            name: 'aten::mean',
            calls: 2,
            host_duration: 376,
            device_duration: 133,
            self_host_duration: 339,
            self_device_duration: 133
          },
          {
            name: 'aten::adaptive_avg_pool2d',
            calls: 2,
            host_duration: 465,
            device_duration: 133,
            self_host_duration: 89,
            self_device_duration: 0
          },
          {
            name: 'aten::_reshape_alias',
            calls: 4,
            host_duration: 170,
            device_duration: 0,
            self_host_duration: 170,
            self_device_duration: 0
          },
          {
            name: 'aten::flatten',
            calls: 2,
            host_duration: 207,
            device_duration: 0,
            self_host_duration: 103,
            self_device_duration: 0
          },
          {
            name: 'aten::transpose',
            calls: 10,
            host_duration: 587,
            device_duration: 0,
            self_host_duration: 465,
            self_device_duration: 0
          },
          {
            name: 'aten::t',
            calls: 10,
            host_duration: 1068,
            device_duration: 0,
            self_host_duration: 481,
            self_device_duration: 0
          },
          {
            name: 'aten::expand',
            calls: 4,
            host_duration: 277,
            device_duration: 0,
            self_host_duration: 227,
            self_device_duration: 0
          },
          {
            name: 'aten::addmm',
            calls: 2,
            host_duration: 809,
            device_duration: 84,
            self_host_duration: 604,
            self_device_duration: 84
          },
          {
            name: 'aten::linear',
            calls: 2,
            host_duration: 1185,
            device_duration: 84,
            self_host_duration: 137,
            self_device_duration: 0
          },
          {
            name: 'aten::_log_softmax',
            calls: 2,
            host_duration: 308,
            device_duration: 14,
            self_host_duration: 271,
            self_device_duration: 14
          },
          {
            name: 'aten::log_softmax',
            calls: 2,
            host_duration: 472,
            device_duration: 14,
            self_host_duration: 153,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_forward',
            calls: 2,
            host_duration: 522,
            device_duration: 8,
            self_host_duration: 476,
            self_device_duration: 8
          },
          {
            name: 'aten::nll_loss',
            calls: 2,
            host_duration: 590,
            device_duration: 8,
            self_host_duration: 68,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_nd',
            calls: 2,
            host_duration: 641,
            device_duration: 8,
            self_host_duration: 51,
            self_device_duration: 0
          },
          {
            name: 'aten::cross_entropy_loss',
            calls: 2,
            host_duration: 1234,
            device_duration: 22,
            self_host_duration: 121,
            self_device_duration: 0
          },
          {
            name: 'aten::fill_',
            calls: 328,
            host_duration: 14541,
            device_duration: 738,
            self_host_duration: 10083,
            self_device_duration: 738
          },
          {
            name: 'aten::ones_like',
            calls: 2,
            host_duration: 516,
            device_duration: 2,
            self_host_duration: 142,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_backward',
            calls: 2,
            host_duration: 573,
            device_duration: 8,
            self_host_duration: 310,
            self_device_duration: 6
          },
          {
            name: 'NllLossBackward0',
            calls: 2,
            host_duration: 774,
            device_duration: 8,
            self_host_duration: 201,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: NllLossBackward0',
            calls: 2,
            host_duration: 1025,
            device_duration: 8,
            self_host_duration: 251,
            self_device_duration: 0
          },
          {
            name: 'aten::_log_softmax_backward_data',
            calls: 2,
            host_duration: 236,
            device_duration: 18,
            self_host_duration: 196,
            self_device_duration: 18
          },
          {
            name: 'LogSoftmaxBackward0',
            calls: 2,
            host_duration: 385,
            device_duration: 18,
            self_host_duration: 149,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: LogSoftmaxBackward0',
            calls: 2,
            host_duration: 632,
            device_duration: 18,
            self_host_duration: 247,
            self_device_duration: 0
          },
          {
            name: 'aten::mm',
            calls: 4,
            host_duration: 668,
            device_duration: 140,
            self_host_duration: 547,
            self_device_duration: 140
          },
          {
            name: 'AddmmBackward0',
            calls: 2,
            host_duration: 1698,
            device_duration: 140,
            self_host_duration: 417,
            self_device_duration: 0
          },
          {
            name: 'aten::sum',
            calls: 2,
            host_duration: 370,
            device_duration: 15,
            self_host_duration: 328,
            self_device_duration: 15
          },
          {
            name: 'autograd::engine::evaluate_function: AddmmBackward0',
            calls: 2,
            host_duration: 2710,
            device_duration: 155,
            self_host_duration: 567,
            self_device_duration: 0
          },
          {
            name: 'torch::autograd::AccumulateGrad',
            calls: 322,
            host_duration: 41184,
            device_duration: 997,
            self_host_duration: 16159,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
            calls: 322,
            host_duration: 70946,
            device_duration: 997,
            self_host_duration: 29762,
            self_device_duration: 0
          },
          {
            name: 'TBackward0',
            calls: 2,
            host_duration: 280,
            device_duration: 0,
            self_host_duration: 64,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: TBackward0',
            calls: 2,
            host_duration: 428,
            device_duration: 0,
            self_host_duration: 148,
            self_device_duration: 0
          },
          {
            name: 'aten::reshape',
            calls: 2,
            host_duration: 170,
            device_duration: 0,
            self_host_duration: 104,
            self_device_duration: 0
          },
          {
            name: 'ReshapeAliasBackward0',
            calls: 2,
            host_duration: 264,
            device_duration: 0,
            self_host_duration: 94,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: ReshapeAliasBackward0',
            calls: 2,
            host_duration: 402,
            device_duration: 0,
            self_host_duration: 138,
            self_device_duration: 0
          },
          {
            name: 'MeanBackward1',
            calls: 2,
            host_duration: 1036,
            device_duration: 75,
            self_host_duration: 231,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: MeanBackward1',
            calls: 2,
            host_duration: 1254,
            device_duration: 75,
            self_host_duration: 218,
            self_device_duration: 0
          },
          {
            name: 'aten::threshold_backward',
            calls: 98,
            host_duration: 13838,
            device_duration: 17984,
            self_host_duration: 12131,
            self_device_duration: 17984
          },
          {
            name: 'ReluBackward0',
            calls: 98,
            host_duration: 21183,
            device_duration: 17984,
            self_host_duration: 7345,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: ReluBackward0',
            calls: 98,
            host_duration: 33492,
            device_duration: 17984,
            self_host_duration: 12309,
            self_device_duration: 0
          },
          {
            name: 'AddBackward0',
            calls: 32,
            host_duration: 251,
            device_duration: 0,
            self_host_duration: 251,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: AddBackward0',
            calls: 32,
            host_duration: 2579,
            device_duration: 0,
            self_host_duration: 2328,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_batch_norm_backward',
            calls: 106,
            host_duration: 62175,
            device_duration: 44433,
            self_host_duration: 36053,
            self_device_duration: 44433
          },
          {
            name: 'CudnnBatchNormBackward0',
            calls: 106,
            host_duration: 69160,
            device_duration: 44433,
            self_host_duration: 6985,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
            calls: 106,
            host_duration: 88613,
            device_duration: 44433,
            self_host_duration: 19453,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_convolution_backward_input',
            calls: 104,
            host_duration: 40820,
            device_duration: 76620,
            self_host_duration: 30768,
            self_device_duration: 76620
          },
          {
            name: 'aten::cudnn_convolution_backward_weight',
            calls: 106,
            host_duration: 44875,
            device_duration: 90108,
            self_host_duration: 27458,
            self_device_duration: 90108
          },
          {
            name: 'aten::cudnn_convolution_backward',
            calls: 106,
            host_duration: 101020,
            device_duration: 166728,
            self_host_duration: 15325,
            self_device_duration: 0
          },
          {
            name: 'CudnnConvolutionBackward0',
            calls: 106,
            host_duration: 107964,
            device_duration: 166728,
            self_host_duration: 6944,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
            calls: 106,
            host_duration: 129129,
            device_duration: 177161,
            self_host_duration: 16746,
            self_device_duration: 0
          },
          {
            name: 'aten::max_pool2d_with_indices_backward',
            calls: 2,
            host_duration: 483,
            device_duration: 3048,
            self_host_duration: 257,
            self_device_duration: 2588
          },
          {
            name: 'MaxPool2DWithIndicesBackward0',
            calls: 2,
            host_duration: 599,
            device_duration: 3048,
            self_host_duration: 116,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
            calls: 2,
            host_duration: 836,
            device_duration: 3048,
            self_host_duration: 237,
            self_device_duration: 0
          },
          {
            name: 'aten::mul_',
            calls: 322,
            host_duration: 23818,
            device_duration: 797,
            self_host_duration: 19073,
            self_device_duration: 797
          }
        ]
      },
      right: {
        name: 'multiple nodes',
        duration: 468427,
        device_duration: 374211,
        total_duration: 644686,
        aggs: [
          {
            name: 'aten::empty',
            calls: 4214,
            host_duration: 31594,
            device_duration: 0,
            self_host_duration: 31594,
            self_device_duration: 0
          },
          {
            name: 'aten::zero_',
            calls: 846,
            host_duration: 6010,
            device_duration: 864,
            self_host_duration: 1910,
            self_device_duration: 0
          },
          {
            name: 'aten::zeros',
            calls: 520,
            host_duration: 10338,
            device_duration: 0,
            self_host_duration: 2951,
            self_device_duration: 0
          },
          {
            name: 'aten::to',
            calls: 2696,
            host_duration: 47031,
            device_duration: 8684,
            self_host_duration: 4258,
            self_device_duration: 0
          },
          {
            name: 'detach',
            calls: 256,
            host_duration: 701,
            device_duration: 0,
            self_host_duration: 698,
            self_device_duration: 0
          },
          {
            name: 'aten::detach',
            calls: 256,
            host_duration: 1374,
            device_duration: 0,
            self_host_duration: 676,
            self_device_duration: 0
          },
          {
            name: 'aten::as_strided',
            calls: 914,
            host_duration: 1013,
            device_duration: 0,
            self_host_duration: 1013,
            self_device_duration: 0
          },
          {
            name: 'aten::unsqueeze',
            calls: 384,
            host_duration: 2074,
            device_duration: 0,
            self_host_duration: 1723,
            self_device_duration: 0
          },
          {
            name: 'aten::empty_strided',
            calls: 1158,
            host_duration: 6859,
            device_duration: 0,
            self_host_duration: 6859,
            self_device_duration: 0
          },
          {
            name: 'aten::copy_',
            calls: 1412,
            host_duration: 25248,
            device_duration: 8684,
            self_host_duration: 16166,
            self_device_duration: 8684
          },
          {
            name: 'aten::_to_copy',
            calls: 1284,
            host_duration: 42773,
            device_duration: 8684,
            self_host_duration: 10227,
            self_device_duration: 0
          },
          {
            name: 'aten::upsample_bilinear2d',
            calls: 128,
            host_duration: 51788,
            device_duration: 0,
            self_host_duration: 46788,
            self_device_duration: 0
          },
          {
            name: 'aten::squeeze',
            calls: 128,
            host_duration: 1035,
            device_duration: 0,
            self_host_duration: 895,
            self_device_duration: 0
          },
          {
            name: 'aten::round',
            calls: 128,
            host_duration: 11074,
            device_duration: 0,
            self_host_duration: 11074,
            self_device_duration: 0
          },
          {
            name: 'aten::slice',
            calls: 260,
            host_duration: 1892,
            device_duration: 0,
            self_host_duration: 1600,
            self_device_duration: 0
          },
          {
            name: 'detach_',
            calls: 512,
            host_duration: 278,
            device_duration: 0,
            self_host_duration: 244,
            self_device_duration: 0
          },
          {
            name: 'aten::detach_',
            calls: 512,
            host_duration: 1341,
            device_duration: 0,
            self_host_duration: 1097,
            self_device_duration: 0
          },
          {
            name: 'aten::result_type',
            calls: 640,
            host_duration: 317,
            device_duration: 0,
            self_host_duration: 317,
            self_device_duration: 0
          },
          {
            name: 'aten::pow',
            calls: 640,
            host_duration: 8857,
            device_duration: 0,
            self_host_duration: 7959,
            self_device_duration: 0
          },
          {
            name: 'aten::sub',
            calls: 640,
            host_duration: 17840,
            device_duration: 0,
            self_host_duration: 7688,
            self_device_duration: 0
          },
          {
            name: 'aten::gt',
            calls: 640,
            host_duration: 6903,
            device_duration: 0,
            self_host_duration: 4901,
            self_device_duration: 0
          },
          {
            name: 'aten::_local_scalar_dense',
            calls: 768,
            host_duration: 395,
            device_duration: 0,
            self_host_duration: 395,
            self_device_duration: 0
          },
          {
            name: 'aten::item',
            calls: 768,
            host_duration: 2532,
            device_duration: 0,
            self_host_duration: 2130,
            self_device_duration: 0
          },
          {
            name: 'aten::is_nonzero',
            calls: 640,
            host_duration: 3601,
            device_duration: 0,
            self_host_duration: 1427,
            self_device_duration: 0
          },
          {
            name: 'aten::div',
            calls: 130,
            host_duration: 11707,
            device_duration: 75,
            self_host_duration: 9531,
            self_device_duration: 75
          },
          {
            name: 'aten::resize_',
            calls: 6,
            host_duration: 79,
            device_duration: 0,
            self_host_duration: 79,
            self_device_duration: 0
          },
          {
            name: 'aten::narrow',
            calls: 4,
            host_duration: 37,
            device_duration: 0,
            self_host_duration: 16,
            self_device_duration: 0
          },
          {
            name: 'aten::_cat',
            calls: 4,
            host_duration: 9241,
            device_duration: 0,
            self_host_duration: 9113,
            self_device_duration: 0
          },
          {
            name: 'aten::cat',
            calls: 4,
            host_duration: 9286,
            device_duration: 0,
            self_host_duration: 45,
            self_device_duration: 0
          },
          {
            name: 'aten::stack',
            calls: 4,
            host_duration: 16195,
            device_duration: 0,
            self_host_duration: 6105,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_convolution',
            calls: 106,
            host_duration: 17357,
            device_duration: 71414,
            self_host_duration: 13601,
            self_device_duration: 71414
          },
          {
            name: 'aten::_convolution',
            calls: 106,
            host_duration: 18514,
            device_duration: 71414,
            self_host_duration: 1157,
            self_device_duration: 0
          },
          {
            name: 'aten::convolution',
            calls: 106,
            host_duration: 19185,
            device_duration: 71414,
            self_host_duration: 671,
            self_device_duration: 0
          },
          {
            name: 'aten::conv2d',
            calls: 106,
            host_duration: 19750,
            device_duration: 71414,
            self_host_duration: 565,
            self_device_duration: 0
          },
          {
            name: 'aten::add',
            calls: 138,
            host_duration: 4973,
            device_duration: 10567,
            self_host_duration: 3157,
            self_device_duration: 10567
          },
          {
            name: 'aten::empty_like',
            calls: 108,
            host_duration: 1924,
            device_duration: 0,
            self_host_duration: 598,
            self_device_duration: 0
          },
          {
            name: 'aten::view',
            calls: 214,
            host_duration: 596,
            device_duration: 0,
            self_host_duration: 596,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_batch_norm',
            calls: 106,
            host_duration: 11083,
            device_duration: 25737,
            self_host_duration: 5031,
            self_device_duration: 25737
          },
          {
            name: 'aten::_batch_norm_impl_index',
            calls: 106,
            host_duration: 11856,
            device_duration: 25737,
            self_host_duration: 773,
            self_device_duration: 0
          },
          {
            name: 'aten::batch_norm',
            calls: 106,
            host_duration: 12386,
            device_duration: 25737,
            self_host_duration: 530,
            self_device_duration: 0
          },
          {
            name: 'aten::clamp_min',
            calls: 98,
            host_duration: 2189,
            device_duration: 12010,
            self_host_duration: 1030,
            self_device_duration: 12010
          },
          {
            name: 'aten::clamp_min_',
            calls: 98,
            host_duration: 2614,
            device_duration: 12010,
            self_host_duration: 425,
            self_device_duration: 0
          },
          {
            name: 'aten::relu_',
            calls: 98,
            host_duration: 3880,
            device_duration: 12010,
            self_host_duration: 1266,
            self_device_duration: 0
          },
          {
            name: 'aten::max_pool2d_with_indices',
            calls: 2,
            host_duration: 112,
            device_duration: 938,
            self_host_duration: 82,
            self_device_duration: 938
          },
          {
            name: 'aten::max_pool2d',
            calls: 2,
            host_duration: 127,
            device_duration: 938,
            self_host_duration: 15,
            self_device_duration: 0
          },
          {
            name: 'aten::add_',
            calls: 998,
            host_duration: 21459,
            device_duration: 13178,
            self_host_duration: 11041,
            self_device_duration: 13178
          },
          {
            name: 'aten::mean',
            calls: 2,
            host_duration: 104,
            device_duration: 126,
            self_host_duration: 76,
            self_device_duration: 126
          },
          {
            name: 'aten::adaptive_avg_pool2d',
            calls: 2,
            host_duration: 117,
            device_duration: 126,
            self_host_duration: 13,
            self_device_duration: 0
          },
          {
            name: 'aten::_reshape_alias',
            calls: 4,
            host_duration: 26,
            device_duration: 0,
            self_host_duration: 26,
            self_device_duration: 0
          },
          {
            name: 'aten::flatten',
            calls: 2,
            host_duration: 31,
            device_duration: 0,
            self_host_duration: 15,
            self_device_duration: 0
          },
          {
            name: 'aten::transpose',
            calls: 10,
            host_duration: 85,
            device_duration: 0,
            self_host_duration: 68,
            self_device_duration: 0
          },
          {
            name: 'aten::t',
            calls: 10,
            host_duration: 145,
            device_duration: 0,
            self_host_duration: 60,
            self_device_duration: 0
          },
          {
            name: 'aten::expand',
            calls: 4,
            host_duration: 30,
            device_duration: 0,
            self_host_duration: 25,
            self_device_duration: 0
          },
          {
            name: 'aten::addmm',
            calls: 2,
            host_duration: 334,
            device_duration: 84,
            self_host_duration: 234,
            self_device_duration: 84
          },
          {
            name: 'aten::linear',
            calls: 2,
            host_duration: 386,
            device_duration: 84,
            self_host_duration: 19,
            self_device_duration: 0
          },
          {
            name: 'aten::_log_softmax',
            calls: 2,
            host_duration: 83,
            device_duration: 14,
            self_host_duration: 55,
            self_device_duration: 14
          },
          {
            name: 'aten::log_softmax',
            calls: 2,
            host_duration: 106,
            device_duration: 14,
            self_host_duration: 20,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_forward',
            calls: 2,
            host_duration: 96,
            device_duration: 8,
            self_host_duration: 68,
            self_device_duration: 8
          },
          {
            name: 'aten::nll_loss',
            calls: 2,
            host_duration: 105,
            device_duration: 8,
            self_host_duration: 9,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_nd',
            calls: 2,
            host_duration: 113,
            device_duration: 8,
            self_host_duration: 8,
            self_device_duration: 0
          },
          {
            name: 'aten::cross_entropy_loss',
            calls: 2,
            host_duration: 243,
            device_duration: 22,
            self_host_duration: 24,
            self_device_duration: 0
          },
          {
            name: 'aten::fill_',
            calls: 328,
            host_duration: 4140,
            device_duration: 866,
            self_host_duration: 1851,
            self_device_duration: 866
          },
          {
            name: 'aten::ones_like',
            calls: 2,
            host_duration: 104,
            device_duration: 2,
            self_host_duration: 14,
            self_device_duration: 0
          },
          {
            name: 'aten::nll_loss_backward',
            calls: 2,
            host_duration: 192,
            device_duration: 9,
            self_host_duration: 84,
            self_device_duration: 6
          },
          {
            name: 'NllLossBackward0',
            calls: 2,
            host_duration: 297,
            device_duration: 9,
            self_host_duration: 105,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: NllLossBackward0',
            calls: 2,
            host_duration: 352,
            device_duration: 9,
            self_host_duration: 55,
            self_device_duration: 0
          },
          {
            name: 'aten::_log_softmax_backward_data',
            calls: 2,
            host_duration: 71,
            device_duration: 18,
            self_host_duration: 43,
            self_device_duration: 18
          },
          {
            name: 'LogSoftmaxBackward0',
            calls: 2,
            host_duration: 91,
            device_duration: 18,
            self_host_duration: 20,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: LogSoftmaxBackward0',
            calls: 2,
            host_duration: 126,
            device_duration: 18,
            self_host_duration: 35,
            self_device_duration: 0
          },
          {
            name: 'aten::mm',
            calls: 4,
            host_duration: 283,
            device_duration: 134,
            self_host_duration: 186,
            self_device_duration: 134
          },
          {
            name: 'AddmmBackward0',
            calls: 2,
            host_duration: 418,
            device_duration: 134,
            self_host_duration: 47,
            self_device_duration: 0
          },
          {
            name: 'aten::sum',
            calls: 2,
            host_duration: 92,
            device_duration: 14,
            self_host_duration: 62,
            self_device_duration: 14
          },
          {
            name: 'autograd::engine::evaluate_function: AddmmBackward0',
            calls: 2,
            host_duration: 594,
            device_duration: 148,
            self_host_duration: 75,
            self_device_duration: 0
          },
          {
            name: 'torch::autograd::AccumulateGrad',
            calls: 322,
            host_duration: 10317,
            device_duration: 1069,
            self_host_duration: 2127,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
            calls: 322,
            host_duration: 15128,
            device_duration: 1069,
            self_host_duration: 4811,
            self_device_duration: 0
          },
          {
            name: 'TBackward0',
            calls: 2,
            host_duration: 30,
            device_duration: 0,
            self_host_duration: 6,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: TBackward0',
            calls: 2,
            host_duration: 45,
            device_duration: 0,
            self_host_duration: 15,
            self_device_duration: 0
          },
          {
            name: 'aten::reshape',
            calls: 2,
            host_duration: 20,
            device_duration: 0,
            self_host_duration: 10,
            self_device_duration: 0
          },
          {
            name: 'ReshapeAliasBackward0',
            calls: 2,
            host_duration: 31,
            device_duration: 0,
            self_host_duration: 11,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: ReshapeAliasBackward0',
            calls: 2,
            host_duration: 48,
            device_duration: 0,
            self_host_duration: 17,
            self_device_duration: 0
          },
          {
            name: 'MeanBackward1',
            calls: 2,
            host_duration: 172,
            device_duration: 75,
            self_host_duration: 18,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: MeanBackward1',
            calls: 2,
            host_duration: 201,
            device_duration: 75,
            self_host_duration: 29,
            self_device_duration: 0
          },
          {
            name: 'aten::threshold_backward',
            calls: 98,
            host_duration: 3652,
            device_duration: 18018,
            self_host_duration: 2361,
            self_device_duration: 18018
          },
          {
            name: 'ReluBackward0',
            calls: 98,
            host_duration: 4567,
            device_duration: 18018,
            self_host_duration: 915,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: ReluBackward0',
            calls: 98,
            host_duration: 6457,
            device_duration: 18018,
            self_host_duration: 1890,
            self_device_duration: 0
          },
          {
            name: 'AddBackward0',
            calls: 32,
            host_duration: 26,
            device_duration: 0,
            self_host_duration: 26,
            self_device_duration: 0
          },
          {
            name: 'autograd::engine::evaluate_function: AddBackward0',
            calls: 32,
            host_duration: 261,
            device_duration: 0,
            self_host_duration: 235,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_batch_norm_backward',
            calls: 106,
            host_duration: 9943,
            device_duration: 44401,
            self_host_duration: 4355,
            self_device_duration: 44401
          },
          {
            name: 'CudnnBatchNormBackward0',
            calls: 106,
            host_duration: 11132,
            device_duration: 44401,
            self_host_duration: 1189,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
            calls: 106,
            host_duration: 14696,
            device_duration: 44401,
            self_host_duration: 3564,
            self_device_duration: 0
          },
          {
            name: 'aten::cudnn_convolution_backward_input',
            calls: 104,
            host_duration: 18813,
            device_duration: 75568,
            self_host_duration: 13997,
            self_device_duration: 75568
          },
          {
            name: 'aten::cudnn_convolution_backward_weight',
            calls: 106,
            host_duration: 18792,
            device_duration: 88992,
            self_host_duration: 11101,
            self_device_duration: 88992
          },
          {
            name: 'aten::cudnn_convolution_backward',
            calls: 106,
            host_duration: 40064,
            device_duration: 164560,
            self_host_duration: 2459,
            self_device_duration: 0
          },
          {
            name: 'CudnnConvolutionBackward0',
            calls: 106,
            host_duration: 41205,
            device_duration: 164560,
            self_host_duration: 1141,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
            calls: 106,
            host_duration: 45209,
            device_duration: 175014,
            self_host_duration: 2826,
            self_device_duration: 0
          },
          {
            name: 'aten::max_pool2d_with_indices_backward',
            calls: 2,
            host_duration: 145,
            device_duration: 3016,
            self_host_duration: 61,
            self_device_duration: 2556
          },
          {
            name: 'MaxPool2DWithIndicesBackward0',
            calls: 2,
            host_duration: 165,
            device_duration: 3016,
            self_host_duration: 20,
            self_device_duration: 0
          },
          {
            name:
              'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
            calls: 2,
            host_duration: 209,
            device_duration: 3016,
            self_host_duration: 44,
            self_device_duration: 0
          },
          {
            name: 'aten::mul_',
            calls: 322,
            host_duration: 6835,
            device_duration: 803,
            self_host_duration: 3630,
            self_device_duration: 803
          }
        ]
      },
      path: '0',
      children: [
        {
          left: {
            name: 'multiple nodes',
            duration: 168,
            device_duration: 0,
            total_duration: 168,
            aggs: [
              {
                name: 'aten::empty',
                calls: 2,
                host_duration: 100,
                device_duration: 0,
                self_host_duration: 100,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 4,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 119,
                device_duration: 0,
                self_host_duration: 64,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'multiple nodes',
            duration: 24,
            device_duration: 0,
            total_duration: 24,
            aggs: [
              {
                name: 'aten::empty',
                calls: 2,
                host_duration: 17,
                device_duration: 0,
                self_host_duration: 17,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 15,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              }
            ]
          },
          path: '0-0'
        },
        {
          left: {
            name: 'enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__',
            duration: 1766103,
            device_duration: 0,
            total_duration: 1766103,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1413,
                host_duration: 62288,
                device_duration: 0,
                self_host_duration: 62288,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 257,
                host_duration: 959,
                device_duration: 0,
                self_host_duration: 959,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 257,
                host_duration: 35273,
                device_duration: 0,
                self_host_duration: 16154,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1344,
                host_duration: 877101,
                device_duration: 0,
                self_host_duration: 18482,
                self_device_duration: 0
              },
              {
                name: 'detach',
                calls: 128,
                host_duration: 2191,
                device_duration: 0,
                self_host_duration: 2191,
                self_device_duration: 0
              },
              {
                name: 'aten::detach',
                calls: 128,
                host_duration: 5301,
                device_duration: 0,
                self_host_duration: 3110,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 450,
                host_duration: 4175,
                device_duration: 0,
                self_host_duration: 4175,
                self_device_duration: 0
              },
              {
                name: 'aten::unsqueeze',
                calls: 192,
                host_duration: 9560,
                device_duration: 0,
                self_host_duration: 8045,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_strided',
                calls: 576,
                host_duration: 24689,
                device_duration: 0,
                self_host_duration: 24689,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 704,
                host_duration: 780214,
                device_duration: 0,
                self_host_duration: 780214,
                self_device_duration: 0
              },
              {
                name: 'aten::_to_copy',
                calls: 640,
                host_duration: 858619,
                device_duration: 0,
                self_host_duration: 53009,
                self_device_duration: 0
              },
              {
                name: 'aten::upsample_bilinear2d',
                calls: 64,
                host_duration: 224031,
                device_duration: 0,
                self_host_duration: 204660,
                self_device_duration: 0
              },
              {
                name: 'aten::squeeze',
                calls: 64,
                host_duration: 4719,
                device_duration: 0,
                self_host_duration: 4119,
                self_device_duration: 0
              },
              {
                name: 'aten::round',
                calls: 64,
                host_duration: 16028,
                device_duration: 0,
                self_host_duration: 16028,
                self_device_duration: 0
              },
              {
                name: 'aten::slice',
                calls: 130,
                host_duration: 8918,
                device_duration: 0,
                self_host_duration: 7569,
                self_device_duration: 0
              },
              {
                name: 'detach_',
                calls: 256,
                host_duration: 2092,
                device_duration: 0,
                self_host_duration: 2092,
                self_device_duration: 0
              },
              {
                name: 'aten::detach_',
                calls: 256,
                host_duration: 7228,
                device_duration: 0,
                self_host_duration: 5136,
                self_device_duration: 0
              },
              {
                name: 'aten::result_type',
                calls: 320,
                host_duration: 884,
                device_duration: 0,
                self_host_duration: 884,
                self_device_duration: 0
              },
              {
                name: 'aten::pow',
                calls: 320,
                host_duration: 43030,
                device_duration: 0,
                self_host_duration: 39068,
                self_device_duration: 0
              },
              {
                name: 'aten::sub',
                calls: 320,
                host_duration: 91440,
                device_duration: 0,
                self_host_duration: 37676,
                self_device_duration: 0
              },
              {
                name: 'aten::gt',
                calls: 320,
                host_duration: 35514,
                device_duration: 0,
                self_host_duration: 24706,
                self_device_duration: 0
              },
              {
                name: 'aten::_local_scalar_dense',
                calls: 384,
                host_duration: 2467,
                device_duration: 0,
                self_host_duration: 2467,
                self_device_duration: 0
              },
              {
                name: 'aten::item',
                calls: 384,
                host_duration: 10375,
                device_duration: 0,
                self_host_duration: 7908,
                self_device_duration: 0
              },
              {
                name: 'aten::is_nonzero',
                calls: 320,
                host_duration: 13905,
                device_duration: 0,
                self_host_duration: 5383,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 64,
                host_duration: 87841,
                device_duration: 0,
                self_host_duration: 76794,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 2,
                host_duration: 117,
                device_duration: 0,
                self_host_duration: 117,
                self_device_duration: 0
              },
              {
                name: 'aten::narrow',
                calls: 2,
                host_duration: 142,
                device_duration: 0,
                self_host_duration: 51,
                self_device_duration: 0
              },
              {
                name: 'aten::_cat',
                calls: 2,
                host_duration: 51526,
                device_duration: 0,
                self_host_duration: 51229,
                self_device_duration: 0
              },
              {
                name: 'aten::cat',
                calls: 2,
                host_duration: 51674,
                device_duration: 0,
                self_host_duration: 148,
                self_device_duration: 0
              },
              {
                name: 'aten::stack',
                calls: 2,
                host_duration: 75677,
                device_duration: 0,
                self_host_duration: 19330,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__',
            duration: 146745,
            device_duration: 0,
            total_duration: 146745,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1413,
                host_duration: 12399,
                device_duration: 0,
                self_host_duration: 12399,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 257,
                host_duration: 98,
                device_duration: 0,
                self_host_duration: 98,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 257,
                host_duration: 7665,
                device_duration: 0,
                self_host_duration: 1689,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1344,
                host_duration: 21137,
                device_duration: 0,
                self_host_duration: 2377,
                self_device_duration: 0
              },
              {
                name: 'detach',
                calls: 128,
                host_duration: 364,
                device_duration: 0,
                self_host_duration: 361,
                self_device_duration: 0
              },
              {
                name: 'aten::detach',
                calls: 128,
                host_duration: 745,
                device_duration: 0,
                self_host_duration: 384,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 450,
                host_duration: 527,
                device_duration: 0,
                self_host_duration: 527,
                self_device_duration: 0
              },
              {
                name: 'aten::unsqueeze',
                calls: 192,
                host_duration: 1050,
                device_duration: 0,
                self_host_duration: 869,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_strided',
                calls: 576,
                host_duration: 3689,
                device_duration: 0,
                self_host_duration: 3689,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 704,
                host_duration: 8695,
                device_duration: 0,
                self_host_duration: 8695,
                self_device_duration: 0
              },
              {
                name: 'aten::_to_copy',
                calls: 640,
                host_duration: 18760,
                device_duration: 0,
                self_host_duration: 6122,
                self_device_duration: 0
              },
              {
                name: 'aten::upsample_bilinear2d',
                calls: 64,
                host_duration: 20349,
                device_duration: 0,
                self_host_duration: 17634,
                self_device_duration: 0
              },
              {
                name: 'aten::squeeze',
                calls: 64,
                host_duration: 562,
                device_duration: 0,
                self_host_duration: 487,
                self_device_duration: 0
              },
              {
                name: 'aten::round',
                calls: 64,
                host_duration: 6658,
                device_duration: 0,
                self_host_duration: 6658,
                self_device_duration: 0
              },
              {
                name: 'aten::slice',
                calls: 130,
                host_duration: 1028,
                device_duration: 0,
                self_host_duration: 870,
                self_device_duration: 0
              },
              {
                name: 'detach_',
                calls: 256,
                host_duration: 142,
                device_duration: 0,
                self_host_duration: 129,
                self_device_duration: 0
              },
              {
                name: 'aten::detach_',
                calls: 256,
                host_duration: 755,
                device_duration: 0,
                self_host_duration: 626,
                self_device_duration: 0
              },
              {
                name: 'aten::result_type',
                calls: 320,
                host_duration: 168,
                device_duration: 0,
                self_host_duration: 168,
                self_device_duration: 0
              },
              {
                name: 'aten::pow',
                calls: 320,
                host_duration: 4922,
                device_duration: 0,
                self_host_duration: 4440,
                self_device_duration: 0
              },
              {
                name: 'aten::sub',
                calls: 320,
                host_duration: 9959,
                device_duration: 0,
                self_host_duration: 4339,
                self_device_duration: 0
              },
              {
                name: 'aten::gt',
                calls: 320,
                host_duration: 3848,
                device_duration: 0,
                self_host_duration: 2737,
                self_device_duration: 0
              },
              {
                name: 'aten::_local_scalar_dense',
                calls: 384,
                host_duration: 209,
                device_duration: 0,
                self_host_duration: 209,
                self_device_duration: 0
              },
              {
                name: 'aten::item',
                calls: 384,
                host_duration: 1398,
                device_duration: 0,
                self_host_duration: 1187,
                self_device_duration: 0
              },
              {
                name: 'aten::is_nonzero',
                calls: 320,
                host_duration: 2013,
                device_duration: 0,
                self_host_duration: 812,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 64,
                host_duration: 7421,
                device_duration: 0,
                self_host_duration: 6234,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 2,
                host_duration: 36,
                device_duration: 0,
                self_host_duration: 36,
                self_device_duration: 0
              },
              {
                name: 'aten::narrow',
                calls: 2,
                host_duration: 19,
                device_duration: 0,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'aten::_cat',
                calls: 2,
                host_duration: 4628,
                device_duration: 0,
                self_host_duration: 4566,
                self_device_duration: 0
              },
              {
                name: 'aten::cat',
                calls: 2,
                host_duration: 4649,
                device_duration: 0,
                self_host_duration: 21,
                self_device_duration: 0
              },
              {
                name: 'aten::stack',
                calls: 2,
                host_duration: 10884,
                device_duration: 0,
                self_host_duration: 5859,
                self_device_duration: 0
              }
            ]
          },
          path: '0-1'
        },
        {
          left: {
            name: 'multiple nodes',
            duration: 5170,
            device_duration: 4402,
            total_duration: 4402,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 2,
                host_duration: 209,
                device_duration: 0,
                self_host_duration: 209,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 2,
                host_duration: 4696,
                device_duration: 4402,
                self_host_duration: 93,
                self_device_duration: 4402
              },
              {
                name: 'aten::_to_copy',
                calls: 2,
                host_duration: 5111,
                device_duration: 4402,
                self_host_duration: 206,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 2,
                host_duration: 5170,
                device_duration: 4402,
                self_host_duration: 59,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'multiple nodes',
            duration: 4681,
            device_duration: 4350,
            total_duration: 4350,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 2,
                host_duration: 65,
                device_duration: 0,
                self_host_duration: 65,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 2,
                host_duration: 4575,
                device_duration: 4350,
                self_host_duration: 26,
                self_device_duration: 4350
              },
              {
                name: 'aten::_to_copy',
                calls: 2,
                host_duration: 4670,
                device_duration: 4350,
                self_host_duration: 30,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 2,
                host_duration: 4681,
                device_duration: 4350,
                self_host_duration: 11,
                self_device_duration: 0
              }
            ]
          },
          path: '0-2'
        },
        {
          left: {
            name: 'nn.Module: ResNet',
            duration: 113664,
            device_duration: 61356,
            total_duration: 61356,
            aggs: [
              {
                name: 'aten::empty',
                calls: 318,
                host_duration: 14161,
                device_duration: 0,
                self_host_duration: 14161,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution',
                calls: 53,
                host_duration: 22091,
                device_duration: 36599,
                self_host_duration: 17567,
                self_device_duration: 36599
              },
              {
                name: 'aten::_convolution',
                calls: 53,
                host_duration: 25744,
                device_duration: 36599,
                self_host_duration: 3653,
                self_device_duration: 0
              },
              {
                name: 'aten::convolution',
                calls: 53,
                host_duration: 27753,
                device_duration: 36599,
                self_host_duration: 2009,
                self_device_duration: 0
              },
              {
                name: 'aten::conv2d',
                calls: 53,
                host_duration: 29777,
                device_duration: 36599,
                self_host_duration: 2024,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 53,
                host_duration: 6519,
                device_duration: 54,
                self_host_duration: 5666,
                self_device_duration: 54
              },
              {
                name: 'aten::empty_like',
                calls: 53,
                host_duration: 5624,
                device_duration: 0,
                self_host_duration: 2390,
                self_device_duration: 0
              },
              {
                name: 'aten::view',
                calls: 53,
                host_duration: 826,
                device_duration: 0,
                self_host_duration: 826,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm',
                calls: 53,
                host_duration: 35818,
                device_duration: 12974,
                self_host_duration: 20557,
                self_device_duration: 12974
              },
              {
                name: 'aten::_batch_norm_impl_index',
                calls: 53,
                host_duration: 38324,
                device_duration: 12974,
                self_host_duration: 2506,
                self_device_duration: 0
              },
              {
                name: 'aten::batch_norm',
                calls: 53,
                host_duration: 40105,
                device_duration: 12974,
                self_host_duration: 1781,
                self_device_duration: 0
              },
              {
                name: 'aten::clamp_min',
                calls: 49,
                host_duration: 2702,
                device_duration: 6002,
                self_host_duration: 1935,
                self_device_duration: 6002
              },
              {
                name: 'aten::clamp_min_',
                calls: 49,
                host_duration: 4273,
                device_duration: 6002,
                self_host_duration: 1571,
                self_device_duration: 0
              },
              {
                name: 'aten::relu_',
                calls: 49,
                host_duration: 8371,
                device_duration: 6002,
                self_host_duration: 4098,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices',
                calls: 1,
                host_duration: 230,
                device_duration: 474,
                self_host_duration: 212,
                self_device_duration: 474
              },
              {
                name: 'aten::max_pool2d',
                calls: 1,
                host_duration: 280,
                device_duration: 474,
                self_host_duration: 50,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 16,
                host_duration: 1546,
                device_duration: 5141,
                self_host_duration: 1290,
                self_device_duration: 5141
              },
              {
                name: 'aten::mean',
                calls: 1,
                host_duration: 189,
                device_duration: 69,
                self_host_duration: 170,
                self_device_duration: 69
              },
              {
                name: 'aten::adaptive_avg_pool2d',
                calls: 1,
                host_duration: 234,
                device_duration: 69,
                self_host_duration: 45,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 52,
                device_duration: 0,
                self_host_duration: 52,
                self_device_duration: 0
              },
              {
                name: 'aten::flatten',
                calls: 1,
                host_duration: 106,
                device_duration: 0,
                self_host_duration: 54,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 2,
                host_duration: 23,
                device_duration: 0,
                self_host_duration: 23,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 1,
                host_duration: 55,
                device_duration: 0,
                self_host_duration: 41,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 1,
                host_duration: 119,
                device_duration: 0,
                self_host_duration: 64,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 49,
                device_duration: 0,
                self_host_duration: 40,
                self_device_duration: 0
              },
              {
                name: 'aten::addmm',
                calls: 1,
                host_duration: 404,
                device_duration: 43,
                self_host_duration: 302,
                self_device_duration: 43
              },
              {
                name: 'aten::linear',
                calls: 1,
                host_duration: 591,
                device_duration: 43,
                self_host_duration: 68,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: ResNet',
            duration: 28725,
            device_duration: 60899,
            total_duration: 60899,
            aggs: [
              {
                name: 'aten::empty',
                calls: 318,
                host_duration: 2292,
                device_duration: 0,
                self_host_duration: 2292,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution',
                calls: 53,
                host_duration: 8713,
                device_duration: 36205,
                self_host_duration: 6819,
                self_device_duration: 36205
              },
              {
                name: 'aten::_convolution',
                calls: 53,
                host_duration: 9298,
                device_duration: 36205,
                self_host_duration: 585,
                self_device_duration: 0
              },
              {
                name: 'aten::convolution',
                calls: 53,
                host_duration: 9653,
                device_duration: 36205,
                self_host_duration: 355,
                self_device_duration: 0
              },
              {
                name: 'aten::conv2d',
                calls: 53,
                host_duration: 9932,
                device_duration: 36205,
                self_host_duration: 279,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 53,
                host_duration: 1897,
                device_duration: 58,
                self_host_duration: 1201,
                self_device_duration: 58
              },
              {
                name: 'aten::empty_like',
                calls: 53,
                host_duration: 933,
                device_duration: 0,
                self_host_duration: 284,
                self_device_duration: 0
              },
              {
                name: 'aten::view',
                calls: 53,
                host_duration: 130,
                device_duration: 0,
                self_host_duration: 130,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm',
                calls: 53,
                host_duration: 5540,
                device_duration: 12913,
                self_host_duration: 2504,
                self_device_duration: 12913
              },
              {
                name: 'aten::_batch_norm_impl_index',
                calls: 53,
                host_duration: 5942,
                device_duration: 12913,
                self_host_duration: 402,
                self_device_duration: 0
              },
              {
                name: 'aten::batch_norm',
                calls: 53,
                host_duration: 6219,
                device_duration: 12913,
                self_host_duration: 277,
                self_device_duration: 0
              },
              {
                name: 'aten::clamp_min',
                calls: 49,
                host_duration: 1108,
                device_duration: 6006,
                self_host_duration: 523,
                self_device_duration: 6006
              },
              {
                name: 'aten::clamp_min_',
                calls: 49,
                host_duration: 1315,
                device_duration: 6006,
                self_host_duration: 207,
                self_device_duration: 0
              },
              {
                name: 'aten::relu_',
                calls: 49,
                host_duration: 1939,
                device_duration: 6006,
                self_host_duration: 624,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices',
                calls: 1,
                host_duration: 53,
                device_duration: 472,
                self_host_duration: 38,
                self_device_duration: 472
              },
              {
                name: 'aten::max_pool2d',
                calls: 1,
                host_duration: 61,
                device_duration: 472,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 16,
                host_duration: 448,
                device_duration: 5140,
                self_host_duration: 268,
                self_device_duration: 5140
              },
              {
                name: 'aten::mean',
                calls: 1,
                host_duration: 53,
                device_duration: 63,
                self_host_duration: 39,
                self_device_duration: 63
              },
              {
                name: 'aten::adaptive_avg_pool2d',
                calls: 1,
                host_duration: 59,
                device_duration: 63,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 8,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::flatten',
                calls: 1,
                host_duration: 15,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 2,
                host_duration: 3,
                device_duration: 0,
                self_host_duration: 3,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 1,
                host_duration: 8,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 1,
                host_duration: 15,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::addmm',
                calls: 1,
                host_duration: 173,
                device_duration: 42,
                self_host_duration: 123,
                self_device_duration: 42
              },
              {
                name: 'aten::linear',
                calls: 1,
                host_duration: 198,
                device_duration: 42,
                self_host_duration: 10,
                self_device_duration: 0
              }
            ]
          },
          path: '0-3'
        },
        {
          left: {
            name: 'nn.Module: CrossEntropyLoss',
            duration: 711,
            device_duration: 11,
            total_duration: 11,
            aggs: [
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax',
                calls: 1,
                host_duration: 158,
                device_duration: 7,
                self_host_duration: 139,
                self_device_duration: 7
              },
              {
                name: 'aten::log_softmax',
                calls: 1,
                host_duration: 241,
                device_duration: 7,
                self_host_duration: 78,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_forward',
                calls: 1,
                host_duration: 256,
                device_duration: 4,
                self_host_duration: 233,
                self_device_duration: 4
              },
              {
                name: 'aten::nll_loss',
                calls: 1,
                host_duration: 290,
                device_duration: 4,
                self_host_duration: 34,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_nd',
                calls: 1,
                host_duration: 313,
                device_duration: 4,
                self_host_duration: 23,
                self_device_duration: 0
              },
              {
                name: 'aten::cross_entropy_loss',
                calls: 1,
                host_duration: 614,
                device_duration: 11,
                self_host_duration: 60,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: CrossEntropyLoss',
            duration: 156,
            device_duration: 11,
            total_duration: 11,
            aggs: [
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 2,
                device_duration: 0,
                self_host_duration: 2,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax',
                calls: 1,
                host_duration: 42,
                device_duration: 7,
                self_host_duration: 28,
                self_device_duration: 7
              },
              {
                name: 'aten::log_softmax',
                calls: 1,
                host_duration: 54,
                device_duration: 7,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 1,
                host_duration: 0,
                device_duration: 0,
                self_host_duration: 0,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_forward',
                calls: 1,
                host_duration: 47,
                device_duration: 4,
                self_host_duration: 34,
                self_device_duration: 4
              },
              {
                name: 'aten::nll_loss',
                calls: 1,
                host_duration: 52,
                device_duration: 4,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_nd',
                calls: 1,
                host_duration: 56,
                device_duration: 4,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::cross_entropy_loss',
                calls: 1,
                host_duration: 119,
                device_duration: 11,
                self_host_duration: 9,
                self_device_duration: 0
              }
            ]
          },
          path: '0-4'
        },
        {
          left: {
            name: 'aten::zeros',
            duration: 119,
            device_duration: 0,
            total_duration: 119,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 47,
                device_duration: 0,
                self_host_duration: 47,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 4,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 119,
                device_duration: 0,
                self_host_duration: 68,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::zeros',
            duration: 17,
            device_duration: 0,
            total_duration: 17,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 8,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 2,
                device_duration: 0,
                self_host_duration: 2,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 17,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              }
            ]
          },
          path: '0-5'
        },
        {
          left: {
            name: 'Optimizer.zero_grad#SGD.zero_grad',
            duration: 22960,
            device_duration: 142,
            total_duration: 142,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 38,
                device_duration: 0,
                self_host_duration: 38,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 161,
                host_duration: 7097,
                device_duration: 142,
                self_host_duration: 4914,
                self_device_duration: 142
              },
              {
                name: 'aten::zero_',
                calls: 161,
                host_duration: 14725,
                device_duration: 142,
                self_host_duration: 7628,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'Optimizer.zero_grad#SGD.zero_grad',
            duration: 4075,
            device_duration: 264,
            total_duration: 264,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 161,
                host_duration: 2036,
                device_duration: 264,
                self_host_duration: 909,
                self_device_duration: 264
              },
              {
                name: 'aten::zero_',
                calls: 161,
                host_duration: 2855,
                device_duration: 264,
                self_host_duration: 819,
                self_device_duration: 0
              }
            ]
          },
          path: '0-6'
        },
        {
          left: {
            name: 'aten::ones_like',
            duration: 253,
            device_duration: 1,
            total_duration: 1,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 1,
                host_duration: 79,
                device_duration: 0,
                self_host_duration: 79,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_like',
                calls: 1,
                host_duration: 126,
                device_duration: 0,
                self_host_duration: 47,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 50,
                device_duration: 1,
                self_host_duration: 35,
                self_device_duration: 1
              },
              {
                name: 'aten::ones_like',
                calls: 1,
                host_duration: 253,
                device_duration: 1,
                self_host_duration: 77,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::ones_like',
            duration: 53,
            device_duration: 1,
            total_duration: 1,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 1,
                host_duration: 18,
                device_duration: 0,
                self_host_duration: 18,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_like',
                calls: 1,
                host_duration: 26,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 20,
                device_duration: 1,
                self_host_duration: 8,
                self_device_duration: 1
              },
              {
                name: 'aten::ones_like',
                calls: 1,
                host_duration: 53,
                device_duration: 1,
                self_host_duration: 7,
                self_device_duration: 0
              }
            ]
          },
          path: '0-7'
        },
        {
          left: {
            name: 'nn.Module: CrossEntropyLoss.backward',
            duration: 898,
            device_duration: 13,
            total_duration: 13,
            aggs: [
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 69,
                device_duration: 1,
                self_host_duration: 43,
                self_device_duration: 1
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 120,
                device_duration: 1,
                self_host_duration: 51,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_backward',
                calls: 1,
                host_duration: 304,
                device_duration: 4,
                self_host_duration: 168,
                self_device_duration: 3
              },
              {
                name: 'NllLossBackward0',
                calls: 1,
                host_duration: 368,
                device_duration: 4,
                self_host_duration: 64,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: NllLossBackward0',
                calls: 1,
                host_duration: 503,
                device_duration: 4,
                self_host_duration: 135,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax_backward_data',
                calls: 1,
                host_duration: 127,
                device_duration: 9,
                self_host_duration: 105,
                self_device_duration: 9
              },
              {
                name: 'LogSoftmaxBackward0',
                calls: 1,
                host_duration: 207,
                device_duration: 9,
                self_host_duration: 80,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: LogSoftmaxBackward0',
                calls: 1,
                host_duration: 349,
                device_duration: 9,
                self_host_duration: 142,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: CrossEntropyLoss.backward',
            duration: 214,
            device_duration: 14,
            total_duration: 14,
            aggs: [
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 36,
                device_duration: 2,
                self_host_duration: 13,
                self_device_duration: 2
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 45,
                device_duration: 2,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_backward',
                calls: 1,
                host_duration: 99,
                device_duration: 5,
                self_host_duration: 43,
                self_device_duration: 3
              },
              {
                name: 'NllLossBackward0',
                calls: 1,
                host_duration: 112,
                device_duration: 5,
                self_host_duration: 13,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: NllLossBackward0',
                calls: 1,
                host_duration: 141,
                device_duration: 5,
                self_host_duration: 29,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax_backward_data',
                calls: 1,
                host_duration: 35,
                device_duration: 9,
                self_host_duration: 21,
                self_device_duration: 9
              },
              {
                name: 'LogSoftmaxBackward0',
                calls: 1,
                host_duration: 46,
                device_duration: 9,
                self_host_duration: 11,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: LogSoftmaxBackward0',
                calls: 1,
                host_duration: 64,
                device_duration: 9,
                self_host_duration: 18,
                self_device_duration: 0
              }
            ]
          },
          path: '0-8'
        },
        {
          left: {
            name: 'nn.Module: ResNet.backward',
            duration: 180998,
            device_duration: 123177,
            total_duration: 123177,
            aggs: [
              {
                name: 'aten::as_strided',
                calls: 5,
                host_duration: 61,
                device_duration: 0,
                self_host_duration: 61,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 4,
                host_duration: 226,
                device_duration: 0,
                self_host_duration: 180,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 4,
                host_duration: 399,
                device_duration: 0,
                self_host_duration: 173,
                self_device_duration: 0
              },
              {
                name: 'aten::mm',
                calls: 2,
                host_duration: 345,
                device_duration: 72,
                self_host_duration: 282,
                self_device_duration: 72
              },
              {
                name: 'AddmmBackward0',
                calls: 1,
                host_duration: 854,
                device_duration: 72,
                self_host_duration: 208,
                self_device_duration: 0
              },
              {
                name: 'aten::sum',
                calls: 1,
                host_duration: 173,
                device_duration: 8,
                self_host_duration: 153,
                self_device_duration: 8
              },
              {
                name: 'aten::view',
                calls: 54,
                host_duration: 971,
                device_duration: 0,
                self_host_duration: 971,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddmmBackward0',
                calls: 1,
                host_duration: 1333,
                device_duration: 80,
                self_host_duration: 271,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 161,
                host_duration: 12621,
                device_duration: 501,
                self_host_duration: 9839,
                self_device_duration: 501
              },
              {
                name: 'torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 20767,
                device_duration: 501,
                self_host_duration: 8146,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 35735,
                device_duration: 501,
                self_host_duration: 14968,
                self_device_duration: 0
              },
              {
                name: 'TBackward0',
                calls: 1,
                host_duration: 128,
                device_duration: 0,
                self_host_duration: 30,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: TBackward0',
                calls: 1,
                host_duration: 197,
                device_duration: 0,
                self_host_duration: 69,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 31,
                device_duration: 0,
                self_host_duration: 31,
                self_device_duration: 0
              },
              {
                name: 'aten::reshape',
                calls: 1,
                host_duration: 79,
                device_duration: 0,
                self_host_duration: 48,
                self_device_duration: 0
              },
              {
                name: 'ReshapeAliasBackward0',
                calls: 1,
                host_duration: 131,
                device_duration: 0,
                self_host_duration: 52,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: ReshapeAliasBackward0',
                calls: 1,
                host_duration: 197,
                device_duration: 0,
                self_host_duration: 66,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 84,
                device_duration: 0,
                self_host_duration: 69,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 1,
                host_duration: 289,
                device_duration: 38,
                self_host_duration: 267,
                self_device_duration: 38
              },
              {
                name: 'MeanBackward1',
                calls: 1,
                host_duration: 489,
                device_duration: 38,
                self_host_duration: 110,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: MeanBackward1',
                calls: 1,
                host_duration: 592,
                device_duration: 38,
                self_host_duration: 103,
                self_device_duration: 0
              },
              {
                name: 'aten::threshold_backward',
                calls: 49,
                host_duration: 6958,
                device_duration: 8972,
                self_host_duration: 6094,
                self_device_duration: 8972
              },
              {
                name: 'ReluBackward0',
                calls: 49,
                host_duration: 10647,
                device_duration: 8972,
                self_host_duration: 3689,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: ReluBackward0',
                calls: 49,
                host_duration: 16826,
                device_duration: 8972,
                self_host_duration: 6179,
                self_device_duration: 0
              },
              {
                name: 'AddBackward0',
                calls: 16,
                host_duration: 129,
                device_duration: 0,
                self_host_duration: 129,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddBackward0',
                calls: 16,
                host_duration: 1301,
                device_duration: 0,
                self_host_duration: 1172,
                self_device_duration: 0
              },
              {
                name: 'aten::empty',
                calls: 370,
                host_duration: 20319,
                device_duration: 0,
                self_host_duration: 20319,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm_backward',
                calls: 53,
                host_duration: 31300,
                device_duration: 22267,
                self_host_duration: 18144,
                self_device_duration: 22267
              },
              {
                name: 'CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 34805,
                device_duration: 22267,
                self_host_duration: 3505,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 44607,
                device_duration: 22267,
                self_host_duration: 9802,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution_backward_input',
                calls: 52,
                host_duration: 20324,
                device_duration: 38733,
                self_host_duration: 15252,
                self_device_duration: 38733
              },
              {
                name: 'aten::cudnn_convolution_backward_weight',
                calls: 53,
                host_duration: 21997,
                device_duration: 45837,
                self_host_duration: 13786,
                self_device_duration: 45837
              },
              {
                name: 'aten::cudnn_convolution_backward',
                calls: 53,
                host_duration: 50059,
                device_duration: 84570,
                self_host_duration: 7738,
                self_device_duration: 0
              },
              {
                name: 'CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 53558,
                device_duration: 84570,
                self_host_duration: 3499,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 64252,
                device_duration: 89775,
                self_host_duration: 8462,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 16,
                host_duration: 2232,
                device_duration: 5205,
                self_host_duration: 1944,
                self_device_duration: 5205
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 61,
                device_duration: 230,
                self_host_duration: 44,
                self_device_duration: 230
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 104,
                device_duration: 230,
                self_host_duration: 43,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices_backward',
                calls: 1,
                host_duration: 246,
                device_duration: 1544,
                self_host_duration: 128,
                self_device_duration: 1314
              },
              {
                name: 'MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 304,
                device_duration: 1544,
                self_host_duration: 58,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 425,
                device_duration: 1544,
                self_host_duration: 121,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: ResNet.backward',
            duration: 43714,
            device_duration: 120604,
            total_duration: 120604,
            aggs: [
              {
                name: 'aten::as_strided',
                calls: 5,
                host_duration: 9,
                device_duration: 0,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 4,
                host_duration: 38,
                device_duration: 0,
                self_host_duration: 31,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 4,
                host_duration: 59,
                device_duration: 0,
                self_host_duration: 21,
                self_device_duration: 0
              },
              {
                name: 'aten::mm',
                calls: 2,
                host_duration: 139,
                device_duration: 67,
                self_host_duration: 90,
                self_device_duration: 67
              },
              {
                name: 'AddmmBackward0',
                calls: 1,
                host_duration: 210,
                device_duration: 67,
                self_host_duration: 23,
                self_device_duration: 0
              },
              {
                name: 'aten::sum',
                calls: 1,
                host_duration: 47,
                device_duration: 7,
                self_host_duration: 32,
                self_device_duration: 7
              },
              {
                name: 'aten::view',
                calls: 54,
                host_duration: 166,
                device_duration: 0,
                self_host_duration: 166,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddmmBackward0',
                calls: 1,
                host_duration: 299,
                device_duration: 74,
                self_host_duration: 37,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 161,
                host_duration: 4087,
                device_duration: 534,
                self_host_duration: 2037,
                self_device_duration: 534
              },
              {
                name: 'torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 5134,
                device_duration: 534,
                self_host_duration: 1047,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 7473,
                device_duration: 534,
                self_host_duration: 2339,
                self_device_duration: 0
              },
              {
                name: 'TBackward0',
                calls: 1,
                host_duration: 14,
                device_duration: 0,
                self_host_duration: 3,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: TBackward0',
                calls: 1,
                host_duration: 21,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::reshape',
                calls: 1,
                host_duration: 10,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'ReshapeAliasBackward0',
                calls: 1,
                host_duration: 14,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: ReshapeAliasBackward0',
                calls: 1,
                host_duration: 21,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 9,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 1,
                host_duration: 70,
                device_duration: 38,
                self_host_duration: 49,
                self_device_duration: 38
              },
              {
                name: 'MeanBackward1',
                calls: 1,
                host_duration: 89,
                device_duration: 38,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: MeanBackward1',
                calls: 1,
                host_duration: 102,
                device_duration: 38,
                self_host_duration: 13,
                self_device_duration: 0
              },
              {
                name: 'aten::threshold_backward',
                calls: 49,
                host_duration: 1789,
                device_duration: 9015,
                self_host_duration: 1158,
                self_device_duration: 9015
              },
              {
                name: 'ReluBackward0',
                calls: 49,
                host_duration: 2237,
                device_duration: 9015,
                self_host_duration: 448,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: ReluBackward0',
                calls: 49,
                host_duration: 3144,
                device_duration: 9015,
                self_host_duration: 907,
                self_device_duration: 0
              },
              {
                name: 'AddBackward0',
                calls: 16,
                host_duration: 12,
                device_duration: 0,
                self_host_duration: 12,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddBackward0',
                calls: 16,
                host_duration: 126,
                device_duration: 0,
                self_host_duration: 114,
                self_device_duration: 0
              },
              {
                name: 'aten::empty',
                calls: 370,
                host_duration: 3292,
                device_duration: 0,
                self_host_duration: 3292,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm_backward',
                calls: 53,
                host_duration: 4896,
                device_duration: 22157,
                self_host_duration: 2136,
                self_device_duration: 22157
              },
              {
                name: 'CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 5495,
                device_duration: 22157,
                self_host_duration: 599,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 7289,
                device_duration: 22157,
                self_host_duration: 1794,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution_backward_input',
                calls: 52,
                host_duration: 9468,
                device_duration: 37714,
                self_host_duration: 7052,
                self_device_duration: 37714
              },
              {
                name: 'aten::cudnn_convolution_backward_weight',
                calls: 53,
                host_duration: 8906,
                device_duration: 44342,
                self_host_duration: 5723,
                self_device_duration: 44342
              },
              {
                name: 'aten::cudnn_convolution_backward',
                calls: 53,
                host_duration: 19611,
                device_duration: 82056,
                self_host_duration: 1237,
                self_device_duration: 0
              },
              {
                name: 'CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 20205,
                device_duration: 82056,
                self_host_duration: 594,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 22185,
                device_duration: 87283,
                self_host_duration: 1386,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 16,
                host_duration: 594,
                device_duration: 5227,
                self_host_duration: 380,
                self_device_duration: 5227
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 24,
                device_duration: 230,
                self_host_duration: 11,
                self_device_duration: 230
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 32,
                device_duration: 230,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices_backward',
                calls: 1,
                host_duration: 72,
                device_duration: 1503,
                self_host_duration: 31,
                self_device_duration: 1273
              },
              {
                name: 'MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 82,
                device_duration: 1503,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 103,
                device_duration: 1503,
                self_host_duration: 21,
                self_device_duration: 0
              }
            ]
          },
          path: '0-9'
        },
        {
          left: {
            name: 'aten::zeros',
            duration: 154,
            device_duration: 0,
            total_duration: 154,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 75,
                device_duration: 0,
                self_host_duration: 75,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 4,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 154,
                device_duration: 0,
                self_host_duration: 75,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::zeros',
            duration: 42,
            device_duration: 0,
            total_duration: 42,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 32,
                device_duration: 0,
                self_host_duration: 32,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 42,
                device_duration: 0,
                self_host_duration: 9,
                self_device_duration: 0
              }
            ]
          },
          path: '0-10'
        },
        {
          left: {
            name: 'Optimizer.step#SGD.step',
            duration: 75880,
            device_duration: 1289,
            total_duration: 1289,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 40,
                device_duration: 0,
                self_host_duration: 40,
                self_device_duration: 0
              },
              {
                name: 'aten::mul_',
                calls: 161,
                host_duration: 11873,
                device_duration: 396,
                self_host_duration: 9505,
                self_device_duration: 396
              },
              {
                name: 'aten::add_',
                calls: 322,
                host_duration: 22327,
                device_duration: 893,
                self_host_duration: 17668,
                self_device_duration: 893
              }
            ]
          },
          right: {
            name: 'Optimizer.step#SGD.step',
            duration: 16441,
            device_duration: 1305,
            total_duration: 1305,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::mul_',
                calls: 161,
                host_duration: 3395,
                device_duration: 399,
                self_host_duration: 1806,
                self_device_duration: 399
              },
              {
                name: 'aten::add_',
                calls: 322,
                host_duration: 6217,
                device_duration: 906,
                self_host_duration: 3246,
                self_device_duration: 906
              }
            ]
          },
          path: '0-11'
        },
        {
          left: {
            name: 'multiple nodes',
            duration: 145,
            device_duration: 0,
            total_duration: 145,
            aggs: [
              {
                name: 'aten::empty',
                calls: 2,
                host_duration: 79,
                device_duration: 0,
                self_host_duration: 79,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 4,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 106,
                device_duration: 0,
                self_host_duration: 62,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'multiple nodes',
            duration: 15,
            device_duration: 0,
            total_duration: 15,
            aggs: [
              {
                name: 'aten::empty',
                calls: 2,
                host_duration: 10,
                device_duration: 0,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 0,
                device_duration: 0,
                self_host_duration: 0,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 9,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              }
            ]
          },
          path: '0-12'
        },
        {
          left: {
            name: 'enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__',
            duration: 1679463,
            device_duration: 0,
            total_duration: 1679463,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1413,
                host_duration: 53837,
                device_duration: 0,
                self_host_duration: 53837,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 257,
                host_duration: 955,
                device_duration: 0,
                self_host_duration: 955,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 257,
                host_duration: 26673,
                device_duration: 0,
                self_host_duration: 16083,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1344,
                host_duration: 824006,
                device_duration: 0,
                self_host_duration: 18525,
                self_device_duration: 0
              },
              {
                name: 'detach',
                calls: 128,
                host_duration: 2188,
                device_duration: 0,
                self_host_duration: 2188,
                self_device_duration: 0
              },
              {
                name: 'aten::detach',
                calls: 128,
                host_duration: 5295,
                device_duration: 0,
                self_host_duration: 3107,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 450,
                host_duration: 4123,
                device_duration: 0,
                self_host_duration: 4123,
                self_device_duration: 0
              },
              {
                name: 'aten::unsqueeze',
                calls: 192,
                host_duration: 9590,
                device_duration: 0,
                self_host_duration: 8097,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_strided',
                calls: 576,
                host_duration: 24764,
                device_duration: 0,
                self_host_duration: 24764,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 704,
                host_duration: 728608,
                device_duration: 0,
                self_host_duration: 728608,
                self_device_duration: 0
              },
              {
                name: 'aten::_to_copy',
                calls: 640,
                host_duration: 805481,
                device_duration: 0,
                self_host_duration: 51350,
                self_device_duration: 0
              },
              {
                name: 'aten::upsample_bilinear2d',
                calls: 64,
                host_duration: 236448,
                device_duration: 0,
                self_host_duration: 216887,
                self_device_duration: 0
              },
              {
                name: 'aten::squeeze',
                calls: 64,
                host_duration: 4682,
                device_duration: 0,
                self_host_duration: 4092,
                self_device_duration: 0
              },
              {
                name: 'aten::round',
                calls: 64,
                host_duration: 15283,
                device_duration: 0,
                self_host_duration: 15283,
                self_device_duration: 0
              },
              {
                name: 'aten::slice',
                calls: 130,
                host_duration: 8844,
                device_duration: 0,
                self_host_duration: 7513,
                self_device_duration: 0
              },
              {
                name: 'detach_',
                calls: 256,
                host_duration: 2102,
                device_duration: 0,
                self_host_duration: 2102,
                self_device_duration: 0
              },
              {
                name: 'aten::detach_',
                calls: 256,
                host_duration: 7286,
                device_duration: 0,
                self_host_duration: 5184,
                self_device_duration: 0
              },
              {
                name: 'aten::result_type',
                calls: 320,
                host_duration: 850,
                device_duration: 0,
                self_host_duration: 850,
                self_device_duration: 0
              },
              {
                name: 'aten::pow',
                calls: 320,
                host_duration: 43219,
                device_duration: 0,
                self_host_duration: 39305,
                self_device_duration: 0
              },
              {
                name: 'aten::sub',
                calls: 320,
                host_duration: 92093,
                device_duration: 0,
                self_host_duration: 37961,
                self_device_duration: 0
              },
              {
                name: 'aten::gt',
                calls: 320,
                host_duration: 35770,
                device_duration: 0,
                self_host_duration: 24869,
                self_device_duration: 0
              },
              {
                name: 'aten::_local_scalar_dense',
                calls: 384,
                host_duration: 2481,
                device_duration: 0,
                self_host_duration: 2481,
                self_device_duration: 0
              },
              {
                name: 'aten::item',
                calls: 384,
                host_duration: 10547,
                device_duration: 0,
                self_host_duration: 8066,
                self_device_duration: 0
              },
              {
                name: 'aten::is_nonzero',
                calls: 320,
                host_duration: 14029,
                device_duration: 0,
                self_host_duration: 5364,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 64,
                host_duration: 79760,
                device_duration: 0,
                self_host_duration: 68841,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 2,
                host_duration: 121,
                device_duration: 0,
                self_host_duration: 121,
                self_device_duration: 0
              },
              {
                name: 'aten::narrow',
                calls: 2,
                host_duration: 138,
                device_duration: 0,
                self_host_duration: 48,
                self_device_duration: 0
              },
              {
                name: 'aten::_cat',
                calls: 2,
                host_duration: 41467,
                device_duration: 0,
                self_host_duration: 41176,
                self_device_duration: 0
              },
              {
                name: 'aten::cat',
                calls: 2,
                host_duration: 41608,
                device_duration: 0,
                self_host_duration: 141,
                self_device_duration: 0
              },
              {
                name: 'aten::stack',
                calls: 2,
                host_duration: 49080,
                device_duration: 0,
                self_host_duration: 2720,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__',
            duration: 123490,
            device_duration: 0,
            total_duration: 123490,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1413,
                host_duration: 6528,
                device_duration: 0,
                self_host_duration: 6528,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 257,
                host_duration: 94,
                device_duration: 0,
                self_host_duration: 94,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 257,
                host_duration: 2448,
                device_duration: 0,
                self_host_duration: 1214,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1344,
                host_duration: 16544,
                device_duration: 0,
                self_host_duration: 1856,
                self_device_duration: 0
              },
              {
                name: 'detach',
                calls: 128,
                host_duration: 337,
                device_duration: 0,
                self_host_duration: 337,
                self_device_duration: 0
              },
              {
                name: 'aten::detach',
                calls: 128,
                host_duration: 629,
                device_duration: 0,
                self_host_duration: 292,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 450,
                host_duration: 464,
                device_duration: 0,
                self_host_duration: 464,
                self_device_duration: 0
              },
              {
                name: 'aten::unsqueeze',
                calls: 192,
                host_duration: 1024,
                device_duration: 0,
                self_host_duration: 854,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_strided',
                calls: 576,
                host_duration: 3009,
                device_duration: 0,
                self_host_duration: 3009,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 704,
                host_duration: 7419,
                device_duration: 0,
                self_host_duration: 7419,
                self_device_duration: 0
              },
              {
                name: 'aten::_to_copy',
                calls: 640,
                host_duration: 14688,
                device_duration: 0,
                self_host_duration: 4039,
                self_device_duration: 0
              },
              {
                name: 'aten::upsample_bilinear2d',
                calls: 64,
                host_duration: 31439,
                device_duration: 0,
                self_host_duration: 29154,
                self_device_duration: 0
              },
              {
                name: 'aten::squeeze',
                calls: 64,
                host_duration: 473,
                device_duration: 0,
                self_host_duration: 408,
                self_device_duration: 0
              },
              {
                name: 'aten::round',
                calls: 64,
                host_duration: 4416,
                device_duration: 0,
                self_host_duration: 4416,
                self_device_duration: 0
              },
              {
                name: 'aten::slice',
                calls: 130,
                host_duration: 864,
                device_duration: 0,
                self_host_duration: 730,
                self_device_duration: 0
              },
              {
                name: 'detach_',
                calls: 256,
                host_duration: 136,
                device_duration: 0,
                self_host_duration: 115,
                self_device_duration: 0
              },
              {
                name: 'aten::detach_',
                calls: 256,
                host_duration: 586,
                device_duration: 0,
                self_host_duration: 471,
                self_device_duration: 0
              },
              {
                name: 'aten::result_type',
                calls: 320,
                host_duration: 149,
                device_duration: 0,
                self_host_duration: 149,
                self_device_duration: 0
              },
              {
                name: 'aten::pow',
                calls: 320,
                host_duration: 3935,
                device_duration: 0,
                self_host_duration: 3519,
                self_device_duration: 0
              },
              {
                name: 'aten::sub',
                calls: 320,
                host_duration: 7881,
                device_duration: 0,
                self_host_duration: 3349,
                self_device_duration: 0
              },
              {
                name: 'aten::gt',
                calls: 320,
                host_duration: 3055,
                device_duration: 0,
                self_host_duration: 2164,
                self_device_duration: 0
              },
              {
                name: 'aten::_local_scalar_dense',
                calls: 384,
                host_duration: 186,
                device_duration: 0,
                self_host_duration: 186,
                self_device_duration: 0
              },
              {
                name: 'aten::item',
                calls: 384,
                host_duration: 1134,
                device_duration: 0,
                self_host_duration: 943,
                self_device_duration: 0
              },
              {
                name: 'aten::is_nonzero',
                calls: 320,
                host_duration: 1588,
                device_duration: 0,
                self_host_duration: 615,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 64,
                host_duration: 4153,
                device_duration: 0,
                self_host_duration: 3203,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 2,
                host_duration: 42,
                device_duration: 0,
                self_host_duration: 42,
                self_device_duration: 0
              },
              {
                name: 'aten::narrow',
                calls: 2,
                host_duration: 18,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::_cat',
                calls: 2,
                host_duration: 4613,
                device_duration: 0,
                self_host_duration: 4547,
                self_device_duration: 0
              },
              {
                name: 'aten::cat',
                calls: 2,
                host_duration: 4637,
                device_duration: 0,
                self_host_duration: 24,
                self_device_duration: 0
              },
              {
                name: 'aten::stack',
                calls: 2,
                host_duration: 5311,
                device_duration: 0,
                self_host_duration: 246,
                self_device_duration: 0
              }
            ]
          },
          path: '0-13'
        },
        {
          left: {
            name: 'multiple nodes',
            duration: 5185,
            device_duration: 4394,
            total_duration: 4394,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 2,
                host_duration: 203,
                device_duration: 0,
                self_host_duration: 203,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 2,
                host_duration: 4687,
                device_duration: 4394,
                self_host_duration: 94,
                self_device_duration: 4394
              },
              {
                name: 'aten::_to_copy',
                calls: 2,
                host_duration: 5113,
                device_duration: 4394,
                self_host_duration: 223,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 2,
                host_duration: 5185,
                device_duration: 4394,
                self_host_duration: 72,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'multiple nodes',
            duration: 4664,
            device_duration: 4334,
            total_duration: 4334,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 2,
                host_duration: 60,
                device_duration: 0,
                self_host_duration: 60,
                self_device_duration: 0
              },
              {
                name: 'aten::copy_',
                calls: 2,
                host_duration: 4559,
                device_duration: 4334,
                self_host_duration: 26,
                self_device_duration: 4334
              },
              {
                name: 'aten::_to_copy',
                calls: 2,
                host_duration: 4655,
                device_duration: 4334,
                self_host_duration: 36,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 2,
                host_duration: 4664,
                device_duration: 4334,
                self_host_duration: 9,
                self_device_duration: 0
              }
            ]
          },
          path: '0-14'
        },
        {
          left: {
            name: 'nn.Module: ResNet',
            duration: 112761,
            device_duration: 59848,
            total_duration: 59848,
            aggs: [
              {
                name: 'aten::empty',
                calls: 318,
                host_duration: 13992,
                device_duration: 0,
                self_host_duration: 13992,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution',
                calls: 53,
                host_duration: 21952,
                device_duration: 35233,
                self_host_duration: 17460,
                self_device_duration: 35233
              },
              {
                name: 'aten::_convolution',
                calls: 53,
                host_duration: 25568,
                device_duration: 35233,
                self_host_duration: 3616,
                self_device_duration: 0
              },
              {
                name: 'aten::convolution',
                calls: 53,
                host_duration: 27534,
                device_duration: 35233,
                self_host_duration: 1966,
                self_device_duration: 0
              },
              {
                name: 'aten::conv2d',
                calls: 53,
                host_duration: 29546,
                device_duration: 35233,
                self_host_duration: 2012,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 53,
                host_duration: 6523,
                device_duration: 53,
                self_host_duration: 5669,
                self_device_duration: 53
              },
              {
                name: 'aten::empty_like',
                calls: 53,
                host_duration: 5605,
                device_duration: 0,
                self_host_duration: 2378,
                self_device_duration: 0
              },
              {
                name: 'aten::view',
                calls: 53,
                host_duration: 829,
                device_duration: 0,
                self_host_duration: 829,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm',
                calls: 53,
                host_duration: 35510,
                device_duration: 12828,
                self_host_duration: 20387,
                self_device_duration: 12828
              },
              {
                name: 'aten::_batch_norm_impl_index',
                calls: 53,
                host_duration: 38030,
                device_duration: 12828,
                self_host_duration: 2520,
                self_device_duration: 0
              },
              {
                name: 'aten::batch_norm',
                calls: 53,
                host_duration: 39727,
                device_duration: 12828,
                self_host_duration: 1697,
                self_device_duration: 0
              },
              {
                name: 'aten::clamp_min',
                calls: 49,
                host_duration: 2715,
                device_duration: 5998,
                self_host_duration: 1950,
                self_device_duration: 5998
              },
              {
                name: 'aten::clamp_min_',
                calls: 49,
                host_duration: 4264,
                device_duration: 5998,
                self_host_duration: 1549,
                self_device_duration: 0
              },
              {
                name: 'aten::relu_',
                calls: 49,
                host_duration: 8337,
                device_duration: 5998,
                self_host_duration: 4073,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices',
                calls: 1,
                host_duration: 212,
                device_duration: 466,
                self_host_duration: 193,
                self_device_duration: 466
              },
              {
                name: 'aten::max_pool2d',
                calls: 1,
                host_duration: 262,
                device_duration: 466,
                self_host_duration: 50,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 16,
                host_duration: 1553,
                device_duration: 5165,
                self_host_duration: 1297,
                self_device_duration: 5165
              },
              {
                name: 'aten::mean',
                calls: 1,
                host_duration: 187,
                device_duration: 64,
                self_host_duration: 169,
                self_device_duration: 64
              },
              {
                name: 'aten::adaptive_avg_pool2d',
                calls: 1,
                host_duration: 231,
                device_duration: 64,
                self_host_duration: 44,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 52,
                device_duration: 0,
                self_host_duration: 52,
                self_device_duration: 0
              },
              {
                name: 'aten::flatten',
                calls: 1,
                host_duration: 101,
                device_duration: 0,
                self_host_duration: 49,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 2,
                host_duration: 21,
                device_duration: 0,
                self_host_duration: 21,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 1,
                host_duration: 51,
                device_duration: 0,
                self_host_duration: 40,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 1,
                host_duration: 120,
                device_duration: 0,
                self_host_duration: 69,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 49,
                device_duration: 0,
                self_host_duration: 39,
                self_device_duration: 0
              },
              {
                name: 'aten::addmm',
                calls: 1,
                host_duration: 405,
                device_duration: 41,
                self_host_duration: 302,
                self_device_duration: 41
              },
              {
                name: 'aten::linear',
                calls: 1,
                host_duration: 594,
                device_duration: 41,
                self_host_duration: 69,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: ResNet',
            duration: 28459,
            device_duration: 59832,
            total_duration: 59832,
            aggs: [
              {
                name: 'aten::empty',
                calls: 318,
                host_duration: 2234,
                device_duration: 0,
                self_host_duration: 2234,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution',
                calls: 53,
                host_duration: 8644,
                device_duration: 35209,
                self_host_duration: 6782,
                self_device_duration: 35209
              },
              {
                name: 'aten::_convolution',
                calls: 53,
                host_duration: 9216,
                device_duration: 35209,
                self_host_duration: 572,
                self_device_duration: 0
              },
              {
                name: 'aten::convolution',
                calls: 53,
                host_duration: 9532,
                device_duration: 35209,
                self_host_duration: 316,
                self_device_duration: 0
              },
              {
                name: 'aten::conv2d',
                calls: 53,
                host_duration: 9818,
                device_duration: 35209,
                self_host_duration: 286,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 53,
                host_duration: 1898,
                device_duration: 55,
                self_host_duration: 1202,
                self_device_duration: 55
              },
              {
                name: 'aten::empty_like',
                calls: 53,
                host_duration: 941,
                device_duration: 0,
                self_host_duration: 300,
                self_device_duration: 0
              },
              {
                name: 'aten::view',
                calls: 53,
                host_duration: 137,
                device_duration: 0,
                self_host_duration: 137,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm',
                calls: 53,
                host_duration: 5543,
                device_duration: 12824,
                self_host_duration: 2527,
                self_device_duration: 12824
              },
              {
                name: 'aten::_batch_norm_impl_index',
                calls: 53,
                host_duration: 5914,
                device_duration: 12824,
                self_host_duration: 371,
                self_device_duration: 0
              },
              {
                name: 'aten::batch_norm',
                calls: 53,
                host_duration: 6167,
                device_duration: 12824,
                self_host_duration: 253,
                self_device_duration: 0
              },
              {
                name: 'aten::clamp_min',
                calls: 49,
                host_duration: 1081,
                device_duration: 6004,
                self_host_duration: 507,
                self_device_duration: 6004
              },
              {
                name: 'aten::clamp_min_',
                calls: 49,
                host_duration: 1299,
                device_duration: 6004,
                self_host_duration: 218,
                self_device_duration: 0
              },
              {
                name: 'aten::relu_',
                calls: 49,
                host_duration: 1941,
                device_duration: 6004,
                self_host_duration: 642,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices',
                calls: 1,
                host_duration: 59,
                device_duration: 466,
                self_host_duration: 44,
                self_device_duration: 466
              },
              {
                name: 'aten::max_pool2d',
                calls: 1,
                host_duration: 66,
                device_duration: 466,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 16,
                host_duration: 443,
                device_duration: 5169,
                self_host_duration: 267,
                self_device_duration: 5169
              },
              {
                name: 'aten::mean',
                calls: 1,
                host_duration: 51,
                device_duration: 63,
                self_host_duration: 37,
                self_device_duration: 63
              },
              {
                name: 'aten::adaptive_avg_pool2d',
                calls: 1,
                host_duration: 58,
                device_duration: 63,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 8,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::flatten',
                calls: 1,
                host_duration: 16,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::as_strided',
                calls: 2,
                host_duration: 3,
                device_duration: 0,
                self_host_duration: 3,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 1,
                host_duration: 10,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 1,
                host_duration: 18,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::addmm',
                calls: 1,
                host_duration: 161,
                device_duration: 42,
                self_host_duration: 111,
                self_device_duration: 42
              },
              {
                name: 'aten::linear',
                calls: 1,
                host_duration: 188,
                device_duration: 42,
                self_host_duration: 9,
                self_device_duration: 0
              }
            ]
          },
          path: '0-15'
        },
        {
          left: {
            name: 'nn.Module: CrossEntropyLoss',
            duration: 712,
            device_duration: 11,
            total_duration: 11,
            aggs: [
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax',
                calls: 1,
                host_duration: 150,
                device_duration: 7,
                self_host_duration: 132,
                self_device_duration: 7
              },
              {
                name: 'aten::log_softmax',
                calls: 1,
                host_duration: 231,
                device_duration: 7,
                self_host_duration: 75,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_forward',
                calls: 1,
                host_duration: 266,
                device_duration: 4,
                self_host_duration: 243,
                self_device_duration: 4
              },
              {
                name: 'aten::nll_loss',
                calls: 1,
                host_duration: 300,
                device_duration: 4,
                self_host_duration: 34,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_nd',
                calls: 1,
                host_duration: 328,
                device_duration: 4,
                self_host_duration: 28,
                self_device_duration: 0
              },
              {
                name: 'aten::cross_entropy_loss',
                calls: 1,
                host_duration: 620,
                device_duration: 11,
                self_host_duration: 61,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: CrossEntropyLoss',
            duration: 156,
            device_duration: 11,
            total_duration: 11,
            aggs: [
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax',
                calls: 1,
                host_duration: 41,
                device_duration: 7,
                self_host_duration: 27,
                self_device_duration: 7
              },
              {
                name: 'aten::log_softmax',
                calls: 1,
                host_duration: 52,
                device_duration: 7,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name: 'aten::resize_',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_forward',
                calls: 1,
                host_duration: 49,
                device_duration: 4,
                self_host_duration: 34,
                self_device_duration: 4
              },
              {
                name: 'aten::nll_loss',
                calls: 1,
                host_duration: 53,
                device_duration: 4,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_nd',
                calls: 1,
                host_duration: 57,
                device_duration: 4,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::cross_entropy_loss',
                calls: 1,
                host_duration: 124,
                device_duration: 11,
                self_host_duration: 15,
                self_device_duration: 0
              }
            ]
          },
          path: '0-16'
        },
        {
          left: {
            name: 'aten::zeros',
            duration: 109,
            device_duration: 0,
            total_duration: 109,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 39,
                device_duration: 0,
                self_host_duration: 39,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 109,
                device_duration: 0,
                self_host_duration: 65,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::zeros',
            duration: 23,
            device_duration: 0,
            total_duration: 23,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 13,
                device_duration: 0,
                self_host_duration: 13,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 23,
                device_duration: 0,
                self_host_duration: 9,
                self_device_duration: 0
              }
            ]
          },
          path: '0-17'
        },
        {
          left: {
            name: 'Optimizer.zero_grad#SGD.zero_grad',
            duration: 24374,
            device_duration: 132,
            total_duration: 132,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 44,
                device_duration: 0,
                self_host_duration: 44,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 161,
                host_duration: 7104,
                device_duration: 132,
                self_host_duration: 4941,
                self_device_duration: 132
              },
              {
                name: 'aten::zero_',
                calls: 161,
                host_duration: 14806,
                device_duration: 132,
                self_host_duration: 7702,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'Optimizer.zero_grad#SGD.zero_grad',
            duration: 4461,
            device_duration: 137,
            total_duration: 137,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 6,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 161,
                host_duration: 1945,
                device_duration: 137,
                self_host_duration: 878,
                self_device_duration: 137
              },
              {
                name: 'aten::zero_',
                calls: 161,
                host_duration: 2805,
                device_duration: 137,
                self_host_duration: 860,
                self_device_duration: 0
              }
            ]
          },
          path: '0-18'
        },
        {
          left: {
            name: 'aten::ones_like',
            duration: 263,
            device_duration: 1,
            total_duration: 1,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 1,
                host_duration: 99,
                device_duration: 0,
                self_host_duration: 99,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_like',
                calls: 1,
                host_duration: 149,
                device_duration: 0,
                self_host_duration: 50,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 49,
                device_duration: 1,
                self_host_duration: 34,
                self_device_duration: 1
              },
              {
                name: 'aten::ones_like',
                calls: 1,
                host_duration: 263,
                device_duration: 1,
                self_host_duration: 65,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::ones_like',
            duration: 51,
            device_duration: 1,
            total_duration: 1,
            aggs: [
              {
                name: 'aten::empty_strided',
                calls: 1,
                host_duration: 18,
                device_duration: 0,
                self_host_duration: 18,
                self_device_duration: 0
              },
              {
                name: 'aten::empty_like',
                calls: 1,
                host_duration: 24,
                device_duration: 0,
                self_host_duration: 6,
                self_device_duration: 0
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 20,
                device_duration: 1,
                self_host_duration: 8,
                self_device_duration: 1
              },
              {
                name: 'aten::ones_like',
                calls: 1,
                host_duration: 51,
                device_duration: 1,
                self_host_duration: 7,
                self_device_duration: 0
              }
            ]
          },
          path: '0-19'
        },
        {
          left: {
            name: 'nn.Module: CrossEntropyLoss.backward',
            duration: 845,
            device_duration: 13,
            total_duration: 13,
            aggs: [
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 58,
                device_duration: 1,
                self_host_duration: 36,
                self_device_duration: 1
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 112,
                device_duration: 1,
                self_host_duration: 54,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_backward',
                calls: 1,
                host_duration: 269,
                device_duration: 4,
                self_host_duration: 142,
                self_device_duration: 3
              },
              {
                name: 'NllLossBackward0',
                calls: 1,
                host_duration: 406,
                device_duration: 4,
                self_host_duration: 137,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: NllLossBackward0',
                calls: 1,
                host_duration: 522,
                device_duration: 4,
                self_host_duration: 116,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax_backward_data',
                calls: 1,
                host_duration: 109,
                device_duration: 9,
                self_host_duration: 91,
                self_device_duration: 9
              },
              {
                name: 'LogSoftmaxBackward0',
                calls: 1,
                host_duration: 178,
                device_duration: 9,
                self_host_duration: 69,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: LogSoftmaxBackward0',
                calls: 1,
                host_duration: 283,
                device_duration: 9,
                self_host_duration: 105,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: CrossEntropyLoss.backward',
            duration: 283,
            device_duration: 13,
            total_duration: 13,
            aggs: [
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 33,
                device_duration: 1,
                self_host_duration: 12,
                self_device_duration: 1
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 41,
                device_duration: 1,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::nll_loss_backward',
                calls: 1,
                host_duration: 93,
                device_duration: 4,
                self_host_duration: 41,
                self_device_duration: 3
              },
              {
                name: 'NllLossBackward0',
                calls: 1,
                host_duration: 185,
                device_duration: 4,
                self_host_duration: 92,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: NllLossBackward0',
                calls: 1,
                host_duration: 211,
                device_duration: 4,
                self_host_duration: 26,
                self_device_duration: 0
              },
              {
                name: 'aten::_log_softmax_backward_data',
                calls: 1,
                host_duration: 36,
                device_duration: 9,
                self_host_duration: 22,
                self_device_duration: 9
              },
              {
                name: 'LogSoftmaxBackward0',
                calls: 1,
                host_duration: 45,
                device_duration: 9,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: LogSoftmaxBackward0',
                calls: 1,
                host_duration: 62,
                device_duration: 9,
                self_host_duration: 17,
                self_device_duration: 0
              }
            ]
          },
          path: '0-20'
        },
        {
          left: {
            name: 'nn.Module: ResNet.backward',
            duration: 180218,
            device_duration: 120676,
            total_duration: 120676,
            aggs: [
              {
                name: 'aten::as_strided',
                calls: 5,
                host_duration: 67,
                device_duration: 0,
                self_host_duration: 67,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 4,
                host_duration: 255,
                device_duration: 0,
                self_host_duration: 204,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 4,
                host_duration: 430,
                device_duration: 0,
                self_host_duration: 175,
                self_device_duration: 0
              },
              {
                name: 'aten::mm',
                calls: 2,
                host_duration: 323,
                device_duration: 68,
                self_host_duration: 265,
                self_device_duration: 68
              },
              {
                name: 'AddmmBackward0',
                calls: 1,
                host_duration: 844,
                device_duration: 68,
                self_host_duration: 209,
                self_device_duration: 0
              },
              {
                name: 'aten::sum',
                calls: 1,
                host_duration: 197,
                device_duration: 7,
                self_host_duration: 175,
                self_device_duration: 7
              },
              {
                name: 'aten::view',
                calls: 54,
                host_duration: 963,
                device_duration: 0,
                self_host_duration: 963,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddmmBackward0',
                calls: 1,
                host_duration: 1377,
                device_duration: 75,
                self_host_duration: 296,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 161,
                host_duration: 12404,
                device_duration: 496,
                self_host_duration: 9659,
                self_device_duration: 496
              },
              {
                name: 'torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 20417,
                device_duration: 496,
                self_host_duration: 8013,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 35211,
                device_duration: 496,
                self_host_duration: 14794,
                self_device_duration: 0
              },
              {
                name: 'TBackward0',
                calls: 1,
                host_duration: 152,
                device_duration: 0,
                self_host_duration: 34,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: TBackward0',
                calls: 1,
                host_duration: 231,
                device_duration: 0,
                self_host_duration: 79,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 35,
                device_duration: 0,
                self_host_duration: 35,
                self_device_duration: 0
              },
              {
                name: 'aten::reshape',
                calls: 1,
                host_duration: 91,
                device_duration: 0,
                self_host_duration: 56,
                self_device_duration: 0
              },
              {
                name: 'ReshapeAliasBackward0',
                calls: 1,
                host_duration: 133,
                device_duration: 0,
                self_host_duration: 42,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: ReshapeAliasBackward0',
                calls: 1,
                host_duration: 205,
                device_duration: 0,
                self_host_duration: 72,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 95,
                device_duration: 0,
                self_host_duration: 79,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 7,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 1,
                host_duration: 324,
                device_duration: 37,
                self_host_duration: 301,
                self_device_duration: 37
              },
              {
                name: 'MeanBackward1',
                calls: 1,
                host_duration: 547,
                device_duration: 37,
                self_host_duration: 121,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: MeanBackward1',
                calls: 1,
                host_duration: 662,
                device_duration: 37,
                self_host_duration: 115,
                self_device_duration: 0
              },
              {
                name: 'aten::threshold_backward',
                calls: 49,
                host_duration: 6880,
                device_duration: 9012,
                self_host_duration: 6037,
                self_device_duration: 9012
              },
              {
                name: 'ReluBackward0',
                calls: 49,
                host_duration: 10536,
                device_duration: 9012,
                self_host_duration: 3656,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: ReluBackward0',
                calls: 49,
                host_duration: 16666,
                device_duration: 9012,
                self_host_duration: 6130,
                self_device_duration: 0
              },
              {
                name: 'AddBackward0',
                calls: 16,
                host_duration: 122,
                device_duration: 0,
                self_host_duration: 122,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddBackward0',
                calls: 16,
                host_duration: 1278,
                device_duration: 0,
                self_host_duration: 1156,
                self_device_duration: 0
              },
              {
                name: 'aten::empty',
                calls: 370,
                host_duration: 21126,
                device_duration: 0,
                self_host_duration: 21126,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm_backward',
                calls: 53,
                host_duration: 30875,
                device_duration: 22166,
                self_host_duration: 17909,
                self_device_duration: 22166
              },
              {
                name: 'CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 34355,
                device_duration: 22166,
                self_host_duration: 3480,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 44006,
                device_duration: 22166,
                self_host_duration: 9651,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution_backward_input',
                calls: 52,
                host_duration: 20496,
                device_duration: 37887,
                self_host_duration: 15516,
                self_device_duration: 37887
              },
              {
                name: 'aten::cudnn_convolution_backward_weight',
                calls: 53,
                host_duration: 22878,
                device_duration: 44271,
                self_host_duration: 13672,
                self_device_duration: 44271
              },
              {
                name: 'aten::cudnn_convolution_backward',
                calls: 53,
                host_duration: 50961,
                device_duration: 82158,
                self_host_duration: 7587,
                self_device_duration: 0
              },
              {
                name: 'CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 54406,
                device_duration: 82158,
                self_host_duration: 3445,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 64877,
                device_duration: 87386,
                self_host_duration: 8284,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 16,
                host_duration: 2187,
                device_duration: 5228,
                self_host_duration: 1909,
                self_device_duration: 5228
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 53,
                device_duration: 230,
                self_host_duration: 36,
                self_device_duration: 230
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 96,
                device_duration: 230,
                self_host_duration: 43,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices_backward',
                calls: 1,
                host_duration: 237,
                device_duration: 1504,
                self_host_duration: 129,
                self_device_duration: 1274
              },
              {
                name: 'MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 295,
                device_duration: 1504,
                self_host_duration: 58,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 411,
                device_duration: 1504,
                self_host_duration: 116,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'nn.Module: ResNet.backward',
            duration: 45132,
            device_duration: 121137,
            total_duration: 121137,
            aggs: [
              {
                name: 'aten::as_strided',
                calls: 5,
                host_duration: 7,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::transpose',
                calls: 4,
                host_duration: 29,
                device_duration: 0,
                self_host_duration: 23,
                self_device_duration: 0
              },
              {
                name: 'aten::t',
                calls: 4,
                host_duration: 53,
                device_duration: 0,
                self_host_duration: 24,
                self_device_duration: 0
              },
              {
                name: 'aten::mm',
                calls: 2,
                host_duration: 144,
                device_duration: 67,
                self_host_duration: 96,
                self_device_duration: 67
              },
              {
                name: 'AddmmBackward0',
                calls: 1,
                host_duration: 208,
                device_duration: 67,
                self_host_duration: 24,
                self_device_duration: 0
              },
              {
                name: 'aten::sum',
                calls: 1,
                host_duration: 45,
                device_duration: 7,
                self_host_duration: 30,
                self_device_duration: 7
              },
              {
                name: 'aten::view',
                calls: 54,
                host_duration: 163,
                device_duration: 0,
                self_host_duration: 163,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddmmBackward0',
                calls: 1,
                host_duration: 295,
                device_duration: 74,
                self_host_duration: 38,
                self_device_duration: 0
              },
              {
                name: 'aten::add_',
                calls: 161,
                host_duration: 4103,
                device_duration: 535,
                self_host_duration: 2037,
                self_device_duration: 535
              },
              {
                name: 'torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 5183,
                device_duration: 535,
                self_host_duration: 1080,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
                calls: 161,
                host_duration: 7655,
                device_duration: 535,
                self_host_duration: 2472,
                self_device_duration: 0
              },
              {
                name: 'TBackward0',
                calls: 1,
                host_duration: 16,
                device_duration: 0,
                self_host_duration: 3,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: TBackward0',
                calls: 1,
                host_duration: 24,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::_reshape_alias',
                calls: 1,
                host_duration: 5,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'aten::reshape',
                calls: 1,
                host_duration: 10,
                device_duration: 0,
                self_host_duration: 5,
                self_device_duration: 0
              },
              {
                name: 'ReshapeAliasBackward0',
                calls: 1,
                host_duration: 17,
                device_duration: 0,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: ReshapeAliasBackward0',
                calls: 1,
                host_duration: 27,
                device_duration: 0,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name: 'aten::expand',
                calls: 1,
                host_duration: 10,
                device_duration: 0,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'aten::to',
                calls: 1,
                host_duration: 1,
                device_duration: 0,
                self_host_duration: 1,
                self_device_duration: 0
              },
              {
                name: 'aten::div',
                calls: 1,
                host_duration: 63,
                device_duration: 37,
                self_host_duration: 45,
                self_device_duration: 37
              },
              {
                name: 'MeanBackward1',
                calls: 1,
                host_duration: 83,
                device_duration: 37,
                self_host_duration: 9,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: MeanBackward1',
                calls: 1,
                host_duration: 99,
                device_duration: 37,
                self_host_duration: 16,
                self_device_duration: 0
              },
              {
                name: 'aten::threshold_backward',
                calls: 49,
                host_duration: 1863,
                device_duration: 9003,
                self_host_duration: 1203,
                self_device_duration: 9003
              },
              {
                name: 'ReluBackward0',
                calls: 49,
                host_duration: 2330,
                device_duration: 9003,
                self_host_duration: 467,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: ReluBackward0',
                calls: 49,
                host_duration: 3313,
                device_duration: 9003,
                self_host_duration: 983,
                self_device_duration: 0
              },
              {
                name: 'AddBackward0',
                calls: 16,
                host_duration: 14,
                device_duration: 0,
                self_host_duration: 14,
                self_device_duration: 0
              },
              {
                name: 'autograd::engine::evaluate_function: AddBackward0',
                calls: 16,
                host_duration: 135,
                device_duration: 0,
                self_host_duration: 121,
                self_device_duration: 0
              },
              {
                name: 'aten::empty',
                calls: 370,
                host_duration: 4638,
                device_duration: 0,
                self_host_duration: 4638,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_batch_norm_backward',
                calls: 53,
                host_duration: 5047,
                device_duration: 22244,
                self_host_duration: 2219,
                self_device_duration: 22244
              },
              {
                name: 'CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 5637,
                device_duration: 22244,
                self_host_duration: 590,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnBatchNormBackward0',
                calls: 53,
                host_duration: 7407,
                device_duration: 22244,
                self_host_duration: 1770,
                self_device_duration: 0
              },
              {
                name: 'aten::cudnn_convolution_backward_input',
                calls: 52,
                host_duration: 9345,
                device_duration: 37854,
                self_host_duration: 6945,
                self_device_duration: 37854
              },
              {
                name: 'aten::cudnn_convolution_backward_weight',
                calls: 53,
                host_duration: 9886,
                device_duration: 44650,
                self_host_duration: 5378,
                self_device_duration: 44650
              },
              {
                name: 'aten::cudnn_convolution_backward',
                calls: 53,
                host_duration: 20453,
                device_duration: 82504,
                self_host_duration: 1222,
                self_device_duration: 0
              },
              {
                name: 'CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 21000,
                device_duration: 82504,
                self_host_duration: 547,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: CudnnConvolutionBackward0',
                calls: 53,
                host_duration: 23024,
                device_duration: 87731,
                self_host_duration: 1440,
                self_device_duration: 0
              },
              {
                name: 'aten::add',
                calls: 16,
                host_duration: 584,
                device_duration: 5227,
                self_host_duration: 374,
                self_device_duration: 5227
              },
              {
                name: 'aten::fill_',
                calls: 1,
                host_duration: 26,
                device_duration: 230,
                self_host_duration: 12,
                self_device_duration: 230
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 33,
                device_duration: 230,
                self_host_duration: 7,
                self_device_duration: 0
              },
              {
                name: 'aten::max_pool2d_with_indices_backward',
                calls: 1,
                host_duration: 73,
                device_duration: 1513,
                self_host_duration: 30,
                self_device_duration: 1283
              },
              {
                name: 'MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 83,
                device_duration: 1513,
                self_host_duration: 10,
                self_device_duration: 0
              },
              {
                name:
                  'autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0',
                calls: 1,
                host_duration: 106,
                device_duration: 1513,
                self_host_duration: 23,
                self_device_duration: 0
              }
            ]
          },
          path: '0-21'
        },
        {
          left: {
            name: 'aten::zeros',
            duration: 160,
            device_duration: 0,
            total_duration: 160,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 87,
                device_duration: 0,
                self_host_duration: 87,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 4,
                device_duration: 0,
                self_host_duration: 4,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 160,
                device_duration: 0,
                self_host_duration: 69,
                self_device_duration: 0
              }
            ]
          },
          right: {
            name: 'aten::zeros',
            duration: 119,
            device_duration: 0,
            total_duration: 119,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 105,
                device_duration: 0,
                self_host_duration: 105,
                self_device_duration: 0
              },
              {
                name: 'aten::zero_',
                calls: 1,
                host_duration: 2,
                device_duration: 0,
                self_host_duration: 2,
                self_device_duration: 0
              },
              {
                name: 'aten::zeros',
                calls: 1,
                host_duration: 119,
                device_duration: 0,
                self_host_duration: 12,
                self_device_duration: 0
              }
            ]
          },
          path: '0-22'
        },
        {
          left: {
            name: 'Optimizer.step#SGD.step',
            duration: 75435,
            device_duration: 1295,
            total_duration: 1295,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 40,
                device_duration: 0,
                self_host_duration: 40,
                self_device_duration: 0
              },
              {
                name: 'aten::mul_',
                calls: 161,
                host_duration: 11945,
                device_duration: 401,
                self_host_duration: 9568,
                self_device_duration: 401
              },
              {
                name: 'aten::add_',
                calls: 322,
                host_duration: 22480,
                device_duration: 894,
                self_host_duration: 17805,
                self_device_duration: 894
              }
            ]
          },
          right: {
            name: 'Optimizer.step#SGD.step',
            duration: 16687,
            device_duration: 1298,
            total_duration: 1298,
            aggs: [
              {
                name: 'aten::empty',
                calls: 1,
                host_duration: 8,
                device_duration: 0,
                self_host_duration: 8,
                self_device_duration: 0
              },
              {
                name: 'aten::mul_',
                calls: 161,
                host_duration: 3440,
                device_duration: 404,
                self_host_duration: 1824,
                self_device_duration: 404
              },
              {
                name: 'aten::add_',
                calls: 322,
                host_duration: 6161,
                device_duration: 894,
                self_host_duration: 3186,
                self_device_duration: 894
              }
            ]
          },
          path: '0-23'
        }
      ]
    })
  }
}
