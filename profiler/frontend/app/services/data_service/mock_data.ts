/** Mock data for tools API */
export const DATA_PLUGIN_PROFILE_TOOLS = {
  'foo': ['overview_page', 'trace_viewer'],
  'empty': [],
  'bar': [
    'overview_page',
    'overview_page@',
    'input_pipeline_analyzer',
    'input_pipeline_analyzer@',
    'memory_viewer',
    'memory_profile#',
    'kernel_stats',
    'op_profile',
    'pod_viewer',
    'trace_viewer',
    'trace_viewer@',
    'trace_viewer^',
    'tensorflow_stats',
    'tensorflow_stats^',
  ]
};

/** Mock data for hosts API */
export const DATA_PLUGIN_PROFILE_HOSTS = ['device', 'host'];

/** Mock data for data API with oveview_page tag */
export const DATA_PLUGIN_PROFILE_OVERVIEW_PAGE_DATA = [
  {
    'cols': [
      {'id': 'selfTimePercent', 'label': 'Time (%)', 'type': 'number'},
      {
        'id': 'cumulativeTimePercent',
        'label': 'Cumulative time (%)',
        'type': 'number'
      },
      {'id': 'category', 'label': 'Category', 'type': 'string'},
      {'id': 'operation', 'label': 'Operation', 'type': 'string'},
      {'id': 'flopRate', 'label': 'GFLOPs/Sec', 'type': 'number'},
    ],
    'rows': [
      {
        'c': [
          {'v': 0.11555},
          {'v': 0.11555},
          {'v': 'InfeedDequeueTuple'},
          {'v': 'InfeedQueue/dequeue'},
          {'v': 0},
        ]
      },
      {
        'c': [
          {'v': 0.03827},
          {'v': 0.15383},
          {'v': ''},
          {'v': 'all-reduce.958'},
          {'v': 1.39586},
        ]
      },
      {
        'c': [
          {'v': 0.01991},
          {'v': 0.17374},
          {'v': 'DepthwiseConv2dNativeBackpropInput'},
          {'v': 'gradients/mnas_v4/DepthwiseConv2dNativeBackpropInput'},
          {'v': 184.05721510136567},
        ]
      },
      {
        'c': [
          {'v': 0.01867},
          {'v': 0.19242},
          {'v': 'Conv2DBackpropInput'},
          {'v': 'gradients/mnas_v4/Conv2DBackpropInput'},
          {'v': 398.40904},
        ]
      },
      {
        'c': [
          {'v': 0.01850},
          {'v': 0.21092},
          {'v': 'Conv2DBackpropInput'},
          {'v': 'gradients/mnas_v4/Conv2DBackpropInput'},
          {'v': 498.64649},
        ]
      },
      {
        'c': [
          {'v': 0.01774},
          {'v': 0.22866},
          {'v': ''},
          {'v': 'reshape.1223'},
          {'v': 0},
        ]
      },
      {
        'c': [
          {'v': 0.01616},
          {'v': 0.24483},
          {'v': 'DepthwiseConv2dNativeBackpropInput'},
          {'v': 'gradients/mnas_v4/DepthwiseConv2dNativeBackpropInput'},
          {'v': 221.49059},
        ]
      },
      {
        'c': [
          {'v': 0.01427},
          {'v': 0.25911},
          {'v': 'DepthwiseConv2dNativeBackpropFilter'},
          {'v': 'gradients/mnas_v4/DepthwiseConv2dNativeBackpropFilter'},
          {'v': 256.82179},
        ]
      },
      {
        'c': [
          {'v': 0.01411},
          {'v': 0.27323},
          {'v': 'Conv2D'},
          {'v': 'mnas_v4/feature_network/stem/conv/Conv2D_Fold'},
          {'v': 471.18847},
        ]
      },
      {
        'c': [
          {'v': 0.01409},
          {'v': 0.28732},
          {'v': 'Conv2D'},
          {'v': 'mnas_v4/feature_network/stem/conv/Conv2D'},
          {'v': 463.72740},
        ]
      },
    ],
    'p': {
      'device_idle_time_percent': '1.1%',
      'flop_rate_utilization_relative_to_roofline': '54.6%',
      'host_idle_time_percent': '59.7%',
      'memory_bw_utilization_relative_to_hw_limit': '54.6%',
      'mxu_utilization_percent': '12.8%',
      'remark_color': '',
      'remark_text': '',
    },
  },
  {
    'cols': [
      {'id': 'stepnum', 'label': 'stepnum', 'type': 'string'},
      {'id': 'computeTimeMs', 'label': 'Compute', 'type': 'number'},
      {'id': 'inputTimeMs', 'label': 'Input', 'type': 'number'},
      {'id': 'idleTimeMs', 'label': 'Idle', 'type': 'number'},
      {
        'id': 'tooltip',
        'label': 'tooltip',
        'type': 'string',
        'p': {'role': 'tooltip'}
      },
    ],
    'rows': [
      {
        'c': [
          {'v': '1491'},
          {'v': 1.2346},
          {'v': 1.3580},
          {'v': 1.4938},
          {
            'v': 'step 1491:\nTime waiting for input data \u003d 1.494 ms, ' +
                'Step time \u003d 2.593 ms'
          },
        ]
      },
      {
        'c': [
          {'v': '1492'},
          {'v': 6.1728},
          {'v': 6.7901},
          {'v': 7.4691},
          {
            'v': 'step 1492:\nTime waiting for input data \u003d 7.469 ms, ' +
                'Step time \u003d 12.963 ms'
          },
        ]
      },
      {
        'c': [
          {'v': '1493'},
          {'v': 3.7037},
          {'v': 4.0741},
          {'v': 4.4815},
          {
            'v': 'step 1493:\nTime waiting for input data \u003d 4.481 ms, ' +
                'Step time \u003d 7.778 ms'
          },
        ]
      },
      {
        'c': [
          {'v': '1494'},
          {'v': 8.6420},
          {'v': 9.5062},
          {'v': 10.4568},
          {
            'v': 'step 1494:\nTime waiting for input data \u003d 10.457 ms, ' +
                'Step time \u003d 18.148 ms'
          },
        ]
      },
      {
        'c': [
          {'v': '1495'},
          {'v': 2.4691},
          {'v': 2.7160},
          {'v': 2.9877},
          {
            'v': 'step 1495:\nTime waiting for input data \u003d 2.988 ms, ' +
                'Step time \u003d 5.185 ms'
          },
        ]
      },
    ],
    'p': {
      'steptime_ms_average': '9.3',
      'steptime_ms_maximum': '18.1',
      'steptime_ms_minimum': '2.6',
      'steptime_ms_standard_deviation': '5.6',
      'idle_ms_average': '5.4',
      'input_ms_average': '4.9',
      'compute_ms_average': '4.4',
    },
  },
  {
    'cols': [],
    'rows': [],
    'p': {
      'device_core_count': '16 (Replica count \u003d 1, num cores per ' +
          'replica \u003d 16)',
      'device_type': 'DragonFish',
      'host_count': '1',
    }
  },
  {
    'cols': [
      {'id': 'tip_type', 'label': 'tip_type', 'type': 'string'},
      {'id': 'link', 'label': 'link', 'type': 'string'},
    ],
    'rows': [
      {
        'c': [
          {'v': 'host'},
          {
            'v': 'Contact <a href="mailto:tf-data-user@gmail.com" target=' +
                '"_blank">tf-data-users@gmail.com</a> with your information.'
          },
        ]
      },
      {
        'c': [
          {'v': 'host'},
          {'v': 'trace_viewer (Use the Trace Viewer.)'},
        ]
      },
      {
        'c': [
          {'v': 'device'},
          {
            'v': 'Contact <a href="mailto:tf-data-user@gmail.com" target=' +
                '"_blank">tf-data-users@gmail.com</a> with your information.'
          },
        ]
      },
      {
        'c': [
          {'v': 'device'},
          {'v': 'trace_viewer (Use the Trace Viewer.)'},
        ]
      },
      {
        'c': [
          {'v': 'doc'},
          {
            'v':
                '<a href="http://tensorflow.org" target="_blank">TensorFlow</a>'
          },
        ]
      },
      {
        'c': [
          {'v': 'doc'},
          {
            'v': '<a href="http://cloud.google.com/tpu/" target="_blank">' +
                'Cloud TPU</a>'
          },
        ]
      },
    ],
    'p': {
      'bottleneck': 'device',
      'statement': '(1) Your program is NOT input-bound because only 0.2% of ' +
          'the total sep time sampled is waiting for input. Therefore, you ' +
          'should focus on reducing the TPU time.',
    }
  },
];

/** Mock data for data API with input_pipeline tag */
export const DATA_PLUGIN_PROFILE_INPUT_PIPELINE_DATA = [
  {
    'cols': [
      {'id': 'stepnum', 'label': 'stepnum', 'type': 'string'},
      {
        'id': 'computeTimeMs',
        'label': 'Compute time (in ms)',
        'type': 'number'
      },
      {'id': 'inputTimeMs', 'label': 'Input time (in ms)', 'type': 'number'},
      {'id': 'idleTimeMs', 'label': 'Idle time (in ms)', 'type': 'number'},
      {
        'id': 'tooltip',
        'label': 'tooltip',
        'type': 'string',
        'p': {'role': 'tooltip'}
      },
      {
        'id': 'infeedPercentAverage',
        'label': '% step time waiting for input data',
        'type': 'number'
      },
      {
        'id': 'infeedPercentMin',
        'label': 'Infeed percent min',
        'type': 'number',
        'p': {'role': 'interval'}
      },
      {
        'id': 'infeedPercentMax',
        'label': 'Infeed percent max',
        'type': 'number',
        'p': {'role': 'interval'}
      },
    ],
    'rows': [
      {
        'c': [
          {'v': '0'},
          {'v': 1.2346},
          {'v': 1.3580},
          {'v': 0.3580},
          {
            'v': 'step 0:\nTime waiting for input data \u003d 1.358 ms, ' +
                'Step time \u003d 1.235 ms'
          },
          {'v': 1.2346},
          {'v': 0.3856},
          {'v': 1.3047},
        ]
      },
      {
        'c': [
          {'v': '1'},
          {'v': 6.1728},
          {'v': 6.7901},
          {'v': 5.7901},
          {
            'v': 'step 1:\nTime waiting for input data \u003d 6.790 ms, ' +
                'Step time \u003d 6.173 ms'
          },
          {'v': 6.1728},
          {'v': 6.0534},
          {'v': 7.2894},
        ]
      },
      {
        'c': [
          {'v': '2'},
          {'v': 3.7037},
          {'v': 4.0741},
          {'v': 3.0741},
          {
            'v': 'step 2:\nTime waiting for input data \u003d 4.074 ms, ' +
                'Step time \u003d 3.704 ms'
          },
          {'v': 3.7037},
          {'v': 2.8989},
          {'v': 4.2162},
        ]
      },
      {
        'c': [
          {'v': '3'},
          {'v': 8.6420},
          {'v': 9.5062},
          {'v': 8.5062},
          {
            'v': 'step 3:\nTime waiting for input data \u003d 9.506 ms, ' +
                'Step time \u003d 8.642 ms'
          },
          {'v': 8.6420},
          {'v': 7.9922},
          {'v': 8.7247},
        ]
      },
      {
        'c': [
          {'v': '4'},
          {'v': 2.4691},
          {'v': 2.7160},
          {'v': 1.7160},
          {
            'v': 'step 4:\nTime waiting for input data \u003d 2.716 ms, ' +
                'Step time \u003d 2.469 ms'
          },
          {'v': 2.4691},
          {'v': 1.6194},
          {'v': 2.9670},
        ]
      },
      {
        'c': [
          {'v': '5'},
          {'v': 1.2346},
          {'v': 1.3580},
          {'v': 0.3580},
          {
            'v': 'step 5:\nTime waiting for input data \u003d 1.358 ms, ' +
                'Step time \u003d 1.235 ms'
          },
          {'v': 1.2346},
          {'v': 0.6609},
          {'v': 2.3396},
        ]
      },
      {
        'c': [
          {'v': '6'},
          {'v': 3.7037},
          {'v': 4.0741},
          {'v': 3.0741},
          {
            'v': 'step 6:\nTime waiting for input data \u003d 4.074 ms, ' +
                'Step time \u003d 3.704 ms'
          },
          {'v': 3.7037},
          {'v': 2.8748},
          {'v': 3.9398},
        ]
      },
      {
        'c': [
          {'v': '7'},
          {'v': 1.2346},
          {'v': 1.3580},
          {'v': 0.3580},
          {
            'v': 'step 7:\nTime waiting for input data \u003d 1.358 ms, ' +
                'Step time \u003d 1.235 ms'
          },
          {'v': 2.4691},
          {'v': 1.8634},
          {'v': 2.9298},
        ]
      },
    ],
    'p': {
      'infeed_percent_average': '3.7',
      'infeed_percent_maximum': '8.6',
      'infeed_percent_minimum': '1.2',
      'infeed_percent_standard_deviation': '2.4',
      'steptime_ms_average': '3.5',
      'steptime_ms_maximum': '8.6',
      'steptime_ms_minimum': '1.2',
      'steptime_ms_standard_deviation': '2.5',
      'input_conclusion': '(1) Your program is NOT input-bound',
      'summary_nextstep':
          'Look at the section 3 for the breakdown of input time on the host.',
    },
  },
  {},
  {
    'cols': [
      {'id': 'opName', 'label': 'Input Op', 'type': 'string'},
      {'id': 'count', 'label': 'Count', 'type': 'number'},
      {'id': 'timeInMs', 'label': 'Total Time (in ms)', 'type': 'number'},
      {
        'id': 'timeInPercent',
        'label': 'Total Time (as % of total input-processing time)',
        'type': 'number'
      },
      {
        'id': 'selfTimeInMs',
        'label': 'Total Self Time (in ms)',
        'type': 'number'
      },
      {
        'id': 'selfTimeInPercent',
        'label': 'Total Self Time (as % of total input-processing time)',
        'type': 'number'
      },
      {'id': 'category', 'label': 'Category', 'type': 'string'},
    ],
    'rows': [
      {
        'c': [
          {'v': 'Iterator::Model::Prefetch::MapAndBatch::Shuffle::SSTable'},
          {'v': 18},
          {'v': 898.126},
          {'v': 0.016},
          {'v': 898.126},
          {'v': 0.016},
          {'v': 'Advanced file read'},
        ]
      },
      {
        'c': [
          {'v': 'Iterator::Model::Prefetch::MapAndBatch::ParallelInterleave'},
          {'v': 374},
          {'v': 831.369},
          {'v': 0.015},
          {'v': 814.603},
          {'v': 0.014},
          {'v': 'Preprocessing'},
        ]
      },
      {
        'c': [
          {'v': 'Iterator::Model::Prefetch::MapAndBatch::Shuffle::SSTable[2]'},
          {'v': 13},
          {'v': 813.236},
          {'v': 0.014},
          {'v': 813.236},
          {'v': 0.014},
          {'v': 'Advanced file read'},
        ]
      },
      {
        'c': [
          {'v': 'input_pipeline_task0/while/InfeedQueue/enqueue/6'},
          {'v': 4},
          {'v': 307.823},
          {'v': 0.005},
          {'v': 307.823},
          {'v': 0.005},
          {'v': 'Enqueue'},
        ]
      },
      {
        'c': [
          {'v': 'Iterator::Model'},
          {'v': 18},
          {'v': 535.884},
          {'v': 0.009},
          {'v': 0.234},
          {'v': 4.245},
          {'v': 'Preprocessing'},
        ]
      },
    ],
    'p': {
      'advanced_file_read_us': '4307208.812',
      'demanded_file_read_us': '0.000',
      'enqueue_us': '3002954.541',
      'preprocessing_us': '180620.171',
      'unclassified_nonequeue_us': '0.000',
    },
  },
  {
    'cols': [{'id': 'link', 'label': 'link', 'type': 'string'}],
    'rows': [
      {
        'c': [{
          'v': 'Enqueuing data: you may want to combine small input data ' +
              'chunks into fewer but larger chunks.'
        }]
      },
      {
        'c': [{
          'v': 'Other data reading or processing: you may consider using the ' +
              '\u003ca href\u003d\u0022https://www.tensorflow.org/programmers' +
              '_guide/datasets\u0022 target\u003d\u0022_blank\u0022\u003eData' +
              'set API\u003c/a\u003e (if you are not using it now)'
        }]
      },
    ],
  },
];

/** Mock data for data API with tensorflow_stats tag */
export const DATA_PLUGIN_PROFILE_TENSORFLOW_STATS_DATA = [
  {
    'cols': [
      {'id': 'rank', 'label': 'Rank', 'type': 'number'},
      {'id': 'host_or_device', 'label': 'Host/device', 'type': 'string'},
      {'id': 'type', 'label': 'Type', 'type': 'string'},
      {'id': 'operation', 'label': 'Operation', 'type': 'string'},
      {'id': 'occurrences', 'label': '#Occurrences', 'type': 'number'},
      {'id': 'total_time', 'label': 'Total time (us)', 'type': 'number'},
      {'id': 'avg_time', 'label': 'Avg. time (us)', 'type': 'number'},
      {
        'id': 'total_self_time',
        'label': 'Total self-time (us)',
        'type': 'number'
      },
      {'id': 'avg_self_time', 'label': 'Avg. self-time (us)', 'type': 'number'},
      {
        'id': 'device_total_self_time_percent',
        'label': 'Total self-time on Device (%)',
        'type': 'number'
      },
      {
        'id': 'device_cumulative_total_self_time_percent',
        'label': 'Cumulative total-self time on Device (%)',
        'type': 'number'
      },
      {
        'id': 'host_total_self_time_percent',
        'label': 'Total self-time on Host (%)',
        'type': 'number'
      },
      {
        'id': 'Host_cumulative_total_self_time_percent',
        'label': 'Cumulative total-self time on Host (%)',
        'type': 'number'
      },
      {
        'id': 'measured_flop_rate',
        'label': 'Measured GFLOPs/Sec',
        'type': 'number'
      },
      {
        'id': 'measured_memory_bw',
        'label': 'Measured Memory BW (GBytes/Sec)',
        'type': 'number'
      },
      {
        'id': 'operational_intensity',
        'label': 'Operational Intensity (FLOPs/Byte)',
        'type': 'number'
      },
      {'id': 'bound_by', 'label': 'Bound by', 'type': 'string'},
    ],
    'p': {
      'architecture_type': 'Convolutional Neural Network (CNN)',
      'device_tf_pprof_link': '',
      'host_tf_pprof_link': '',
      'task_type': 'Training',
    },
    'rows': [
      {
        'c': [
          {'v': 1}, {'v': 'Device'}, {'v': 'Conv2D'}, {'v': 'Conv2D_Fold'},
          {'v': 50}, {'v': 4000000.5}, {'v': 80000.01}, {'v': 4000000.5},
          {'v': 80000.01}, {'v': 0.4}, {'v': 0.4}, {'v': 0}, {'v': 0},
          {'v': 5413.5}, {'v': 85}, {'v': 63.7}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 2}, {'v': 'Device'}, {'v': 'Conv2D'}, {'v': 'Conv2D'},
          {'v': 40}, {'v': 1200000.5}, {'v': 30000.01}, {'v': 1200000.5},
          {'v': 30000.01}, {'v': 0.12}, {'v': 0.52}, {'v': 0}, {'v': 0},
          {'v': 447.2}, {'v': 121.5}, {'v': 3.7}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 3}, {'v': 'Device'}, {'v': 'DepthwiseConv2dNative'},
          {'v': 'DepthwiseConv2dNative'}, {'v': 25}, {'v': 2000000.5},
          {'v': 80000.02}, {'v': 2000000.5}, {'v': 80000.02}, {'v': 0.2},
          {'v': 0.72}, {'v': 0}, {'v': 0}, {'v': 5383.2}, {'v': 62.8},
          {'v': 85.8}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 4}, {'v': 'Device'}, {'v': 'Max'}, {'v': 'BatchMax'}, {'v': 10},
          {'v': 1300000.5}, {'v': 130000.05}, {'v': 1300000.5},
          {'v': 130000.05}, {'v': 0.13}, {'v': 0.85}, {'v': 0}, {'v': 0},
          {'v': 111.6}, {'v': 210.5}, {'v': 0.5}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 5}, {'v': 'Device'}, {'v': 'IDLE'}, {'v': 'IDLE'}, {'v': 10},
          {'v': 1500000.5}, {'v': 150000.05}, {'v': 1500000.5},
          {'v': 150000.05}, {'v': 0.15}, {'v': 1}, {'v': 0}, {'v': 0}, {'v': 0},
          {'v': 0}, {'v': 0}, {'v': 'Unknown'}
        ]
      },
      {
        'c': [
          {'v': 6}, {'v': 'Host'}, {'v': 'IDLE'}, {'v': 'IDLE'}, {'v': 30},
          {'v': 4500000.1}, {'v': 1500000}, {'v': 4500000.1}, {'v': 1500000},
          {'v': 0}, {'v': 0}, {'v': 0.45}, {'v': 0.45}, {'v': 0}, {'v': 0},
          {'v': 0}, {'v': 'Unknown'}
        ]
      },
      {
        'c': [
          {'v': 7}, {'v': 'Host'}, {'v': 'OutfeedDequeueTuple'},
          {'v': 'OutfeedDequeueTuple'}, {'v': 20}, {'v': 3000000.1},
          {'v': 150000}, {'v': 3000000.1}, {'v': 150000}, {'v': 0}, {'v': 0},
          {'v': 0.30}, {'v': 0.75}, {'v': 0}, {'v': 0}, {'v': 0},
          {'v': 'Unknown'}
        ]
      },
      {
        'c': [
          {'v': 8}, {'v': 'Host'}, {'v': 'OutfeedDequeueTuple'},
          {'v': 'OutfeedDequeueTuple_1'}, {'v': 10}, {'v': 2500000.1},
          {'v': 250000.01}, {'v': 2500000.1}, {'v': 250000.01}, {'v': 0},
          {'v': 0}, {'v': 0.25}, {'v': 1}, {'v': 0}, {'v': 0}, {'v': 0},
          {'v': 'Unknown'}
        ]
      },
    ]
  },
  {
    'cols': [
      {'id': 'rank', 'label': 'Rank', 'type': 'number'},
      {'id': 'host_or_device', 'label': 'Host/device', 'type': 'string'},
      {'id': 'type', 'label': 'Type', 'type': 'string'},
      {'id': 'operation', 'label': 'Operation', 'type': 'string'},
      {'id': 'occurrences', 'label': '#Occurrences', 'type': 'number'},
      {'id': 'total_time', 'label': 'Total time (us)', 'type': 'number'},
      {'id': 'avg_time', 'label': 'Avg. time (us)', 'type': 'number'},
      {
        'id': 'total_self_time',
        'label': 'Total self-time (us)',
        'type': 'number'
      },
      {'id': 'avg_self_time', 'label': 'Avg. self-time (us)', 'type': 'number'},
      {
        'id': 'device_total_self_time_percent',
        'label': 'Total self-time on Device (%)',
        'type': 'number'
      },
      {
        'id': 'device_cumulative_total_self_time_percent',
        'label': 'Cumulative total-self time on Device (%)',
        'type': 'number'
      },
      {
        'id': 'host_total_self_time_percent',
        'label': 'Total self-time on Host (%)',
        'type': 'number'
      },
      {
        'id': 'Host_cumulative_total_self_time_percent',
        'label': 'Cumulative total-self time on Host (%)',
        'type': 'number'
      },
      {
        'id': 'measured_flop_rate',
        'label': 'Measured GFLOPs/Sec',
        'type': 'number'
      },
      {
        'id': 'measured_memory_bw',
        'label': 'Measured Memory BW (GBytes/Sec)',
        'type': 'number'
      },
      {
        'id': 'operational_intensity',
        'label': 'Operational Intensity (FLOPs/Byte)',
        'type': 'number'
      },
      {'id': 'bound_by', 'label': 'Bound by', 'type': 'string'},
    ],
    'p': {
      'architecture_type': 'Convolutional Neural Network (CNN)',
      'device_tf_pprof_link': '',
      'host_tf_pprof_link': '',
      'task_type': 'Training',
    },
    'rows': [
      {
        'c': [
          {'v': 1}, {'v': 'Device'}, {'v': 'Conv2D'}, {'v': 'Conv2D_Fold'},
          {'v': 50}, {'v': 4000000.5}, {'v': 80000.01}, {'v': 4000000.5},
          {'v': 80000.01}, {'v': 0.46}, {'v': 0.46}, {'v': 0}, {'v': 0},
          {'v': 5413.5}, {'v': 85}, {'v': 63.7}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 2}, {'v': 'Device'}, {'v': 'Conv2D'}, {'v': 'Conv2D'},
          {'v': 40}, {'v': 1200000.5}, {'v': 30000.01}, {'v': 1200000.5},
          {'v': 30000.01}, {'v': 0.14}, {'v': 0.6}, {'v': 0}, {'v': 0},
          {'v': 447.2}, {'v': 121.5}, {'v': 3.7}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 3}, {'v': 'Device'}, {'v': 'DepthwiseConv2dNative'},
          {'v': 'DepthwiseConv2dNative'}, {'v': 25}, {'v': 2000000.5},
          {'v': 80000.02}, {'v': 2000000.5}, {'v': 80000.02}, {'v': 0.24},
          {'v': 0.84}, {'v': 0}, {'v': 0}, {'v': 5383.2}, {'v': 62.8},
          {'v': 85.8}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 4}, {'v': 'Device'}, {'v': 'Max'}, {'v': 'BatchMax'}, {'v': 10},
          {'v': 1300000.5}, {'v': 130000.05}, {'v': 1300000.5},
          {'v': 130000.05}, {'v': 0.16}, {'v': 1}, {'v': 0}, {'v': 0},
          {'v': 111.6}, {'v': 210.5}, {'v': 0.5}, {'v': 'Memory'}
        ]
      },
      {
        'c': [
          {'v': 5}, {'v': 'Host'}, {'v': 'OutfeedDequeueTuple'},
          {'v': 'OutfeedDequeueTuple'}, {'v': 20}, {'v': 3000000.1},
          {'v': 150000}, {'v': 3000000.1}, {'v': 150000}, {'v': 0}, {'v': 0},
          {'v': 0.67}, {'v': 0.67}, {'v': 0}, {'v': 0}, {'v': 0},
          {'v': 'Unknown'}
        ]
      },
      {
        'c': [
          {'v': 6}, {'v': 'Host'}, {'v': 'OutfeedDequeueTuple'},
          {'v': 'OutfeedDequeueTuple_1'}, {'v': 10}, {'v': 2500000.1},
          {'v': 250000.01}, {'v': 2500000.1}, {'v': 250000.01}, {'v': 0},
          {'v': 0}, {'v': 0.33}, {'v': 1}, {'v': 0}, {'v': 0}, {'v': 0},
          {'v': 'Unknown'}
        ]
      },
    ]
  },
];

/* tslint:disable no-any */
/** Mock data for data API with memory_profile tag */
export const DATA_PLUGIN_PROFILE_MEMORY_PROFILE_DATA = {
  'memoryIds': ['0', '1', '2'],
  'memoryProfilePerAllocator': {
    '0': {
      'activeAllocations': [
        {'numOccurrences': '1', 'snapshotIndex': '-1', 'specialIndex': '0'},
        {'numOccurrences': '1', 'snapshotIndex': '-2', 'specialIndex': '1'},
        {'numOccurrences': '1', 'snapshotIndex': '2', 'specialIndex': '-1'}
      ],
      'memoryProfileSnapshots': [
        {
          'activityMetadata': {
            'address': '222333',
            'allocationBytes': '256',
            'dataType': 'float',
            'memoryActivity': 'ALLOCATION' as any,
            'regionType': 'output',
            'requestedBytes': '200',
            'stepId': '0',
            'tensorShape': '[3, 3, 512, 512]',
            'tfOpName': 'foo/bar'
          },
          'aggregationStats': {
            'fragmentation': 0.1,
            'freeMemoryBytes': '5000',
            'heapAllocatedBytes': '3000',
            'peakBytesInUse': '8500',
            'stackReservedBytes': '2000'
          },
          'timeOffsetPs': '40000000'
        },
        {
          'activityMetadata': {
            'address': '222333',
            'allocationBytes': '256',
            'dataType': 'float',
            'memoryActivity': 'DEALLOCATION' as any,
            'regionType': 'output',
            'requestedBytes': '200',
            'stepId': '0',
            'tensorShape': '[3, 3, 512, 512]',
            'tfOpName': 'foo/bar'
          },
          'aggregationStats': {
            'fragmentation': 0.08,
            'freeMemoryBytes': '5256',
            'heapAllocatedBytes': '2744',
            'peakBytesInUse': '8500',
            'stackReservedBytes': '2000'
          },
          'timeOffsetPs': '50000000'
        },
        {
          'activityMetadata': {
            'address': '345678',
            'allocationBytes': '300',
            'dataType': 'int64',
            'memoryActivity': 'ALLOCATION',
            'regionType': 'temp',
            'requestedBytes': '300',
            'stepId': '0',
            'tensorShape': '[]',
            'tfOpName': 'mul_grad/Sum'
          },
          'aggregationStats': {
            'fragmentation': 0.8,
            'freeMemoryBytes': '3000',
            'heapAllocatedBytes': '5000',
            'peakBytesInUse': '9500',
            'stackReservedBytes': '2000'
          },
          'timeOffsetPs': '70000000'
        }
      ],
      'profileSummary': {
        'memoryCapacity': '10000',
        'peakBytesUsageLifetime': '9500',
        'peakStats': {
          'fragmentation': 0.8,
          'freeMemoryBytes': '3000',
          'heapAllocatedBytes': '5000',
          'peakBytesInUse': '7000',
          'stackReservedBytes': '2000'
        },
        'peakStatsTimePs': '70000000'
      },
      'specialAllocations': [
        {
          'address': '0',
          'allocationBytes': '4700',
          'dataType': 'INVALID',
          'memoryActivity': 'ALLOCATION' as any,
          'regionType': 'persist',
          'requestedBytes': '4700',
          'stepId': '0',
          'tensorShape': 'unknown',
          'tfOpName': 'preallocated/unknown'
        },
        {
          'address': '0',
          'allocationBytes': '2000',
          'dataType': 'INVALID',
          'memoryActivity': 'ALLOCATION',
          'regionType': 'stack',
          'requestedBytes': '2000',
          'stepId': '0',
          'tensorShape': 'unknown',
          'tfOpName': 'stack'
        }
      ]
    }
  },
  'numHosts': 1
};

/** Mock data for data API with memory_viewer tag */
export const DATA_PLUGIN_PROFILE_MEMORY_VIEWER_DATA = {
  'hloModule': {
    'name': 'Good Module',
    'entryComputationName': 'comp',
    'entryComputationId': '0',
    'computations': [
      {
        'name': 'comp',
        'instructions': [
          {
            'name': 'ins-0',
            'opcode': 'fusion',
            'shape': {
              'elementType': 'F32' as any,
              'layout': {
                'format': 'DENSE' as any,
                'maxSparseElements': '0',
              },
            },
            'id': '0',
          },
          {
            'name': 'ins-1',
            'opcode': 'fusion',
            'shape': {
              'elementType': 'F32' as any,
              'dimensions': ['32', '16', '16', '32'],
              'layout': {
                'format': 'DENSE' as any,
                'minorToMajor': ['3', '0', '2', '1'],
                'maxSparseElements': '0',
              },
            },
            'metadata': {
              'opType': 'Relu',
              'opName': 'sequentila/conv2d/Relu',
              'sourceLine': 0,
            },
            'id': '1',
          },
          {
            'name': 'ins-2',
            'opcode': 'fusion',
            'shape': {
              'elementType': 'F32',
              'layout': {
                'format': 'DENSE',
                'maxSparseElements': '0',
              },
            },
            'id': '2',
          },
          {
            'name': 'ins-3',
            'opcode': 'fusion',
            'shape': {
              'elementType': 'F32',
              'layout': {
                'format': 'DENSE',
                'maxSparseElements': '0',
              },
            },
            'id': '3',
          },
          {
            'name': 'ins-4',
            'opcode': 'fusion',
            'shape': {
              'elementType': 'F32',
              'layout': {
                'format': 'DENSE',
                'maxSparseElements': '0',
              },
            },
            'id': '4',
          },
        ],
        'id': '0',
      },
    ],
  },
  'bufferAssignment': {
    'logicalBuffers': [
      {
        'id': '0',
        'size': '1048576',
        'definedAt': {
          'computationName': 'comp',
          'instructionName': 'ins-0',
        },
      },
      {
        'id': '1',
        'size': '2097512',
        'definedAt': {
          'computationName': 'comp',
          'instructionName': 'ins-1',
        },
      },
      {
        'id': '2',
        'size': '3145728',
        'definedAt': {
          'computationName': 'comp',
          'instructionName': 'ins-2',
        },
      },
      {
        'id': '3',
        'size': '5242880',
        'definedAt': {
          'computationName': 'comp',
          'instructionName': 'ins-3',
        },
      },
      {
        'id': '4',
        'size': '1048576',
        'definedAt': {
          'computationName': 'comp',
          'instructionName': 'ins-4',
        },
      },
    ],
    'bufferAllocations': [
      {
        'index': '0',
        'size': '10485676',
        'isThreadLocal': true,
        'isResuable': false,
        'isEntryComputationParameter': false,
        'maybeLiveOut': false,
        'assigned': [{
          'logicalBufferId': '0',
          'offset': '0',
          'size': '1048576',
        }],
      },
      {
        'index': '1',
        'size': '2097512',
        'isThreadLocal': true,
        'isResuable': false,
        'isEntryComputationParameter': false,
        'maybeLiveOut': false,
        'assigned': [{
          'logicalBufferId': '1',
          'offset': '0',
          'size': '2097512',
        }],
      },
      {
        'index': '2',
        'size': '3145728',
        'isThreadLocal': true,
        'isResuable': false,
        'isEntryComputationParameter': false,
        'maybeLiveOut': false,
        'assigned': [{
          'logicalBufferId': '2',
          'offset': '0',
          'size': '3145728',
        }],
      },
      {
        'index': '3',
        'size': '5242880',
        'isThreadLocal': true,
        'isResuable': false,
        'isEntryComputationParameter': false,
        'maybeLiveOut': false,
        'assigned': [{
          'logicalBufferId': '3',
          'offset': '0',
          'size': '5242880',
        }],
      },
      {
        'index': '4',
        'size': '10485676',
        'isThreadLocal': true,
        'isResuable': false,
        'isEntryComputationParameter': false,
        'maybeLiveOut': false,
        'assigned': [{
          'logicalBufferId': '4',
          'offset': '0',
          'size': '1048576',
        }],
      },
    ],
    'heapSimulatorTraces': [
      {
        'events': [
          {
            'kind': 'ALLOC' as any,
            'bufferId': '0',
            'computationName': 'comp',
            'instructionName': 'ins-0',
          },
          {
            'kind': 'ALLOC' as any,
            'bufferId': '1',
            'computationName': 'comp',
            'instructionName': 'ins-1',
          },
          {
            'kind': 'ALLOC',
            'bufferId': '2',
            'computationName': 'comp',
            'instructionName': 'ins-2',
          },
          {
            'kind': 'ALLOC',
            'bufferId': '3',
            'computationName': 'comp',
            'instructionName': 'ins-3',
          },
          {
            'kind': 'FREE',
            'bufferId': '3',
            'computationName': 'comp',
            'instructionName': 'ins-3',
          },
          {
            'kind': 'FREE',
            'bufferId': '2',
            'computationName': 'comp',
            'instructionName': 'ins-2',
          },
          {
            'kind': 'FREE',
            'bufferId': '1',
            'computationName': 'comp',
            'instructionName': 'ins-1',
          },
          {
            'kind': 'ALLOC',
            'bufferId': '4',
            'computationName': 'comp',
            'instructionName': 'ins-4',
          },
          {
            'kind': 'FREE',
            'bufferId': '4',
            'computationName': 'comp',
            'instructionName': 'ins-4',
          },
          {
            'kind': 'FREE',
            'bufferId': '0',
            'computationName': 'comp',
            'instructionName': 'ins-0',
          },
        ],
      },
    ],
  },
};
/* tslint:enable */

/** Mock data for data API with op_profile tag */
export const DATA_PLUGIN_PROFILE_OP_PROFILE_DATA = {
  'byCategory': {
    'name': 'byCategory',
    'metrics': {
      'time': 1,
      'flops': 0.4,
      'memoryBandwidth': 0.05,
    },
    'children': [
      {
        'name': 'Good ops',
        'metrics': {
          'time': 0.6,
          'flops': 0.4,
        },
        'category': {},
        'children': [
          {
            'name': '%convolution',
            'metrics': {
              'time': 0.2,
              'flops': 0.18,
            },
            'xla': {
              'op': 'convolution',
              'expression': '%convolution = something',
              'provenance': 'Convolution2D',
              'category': 'Good ops',
              'layout': {
                'dimensions': [
                  {
                    'size': 200,
                    'alignment': 128,
                    'semantics': 'feature',
                  },
                  {
                    'size': 8,
                    'alignment': 8,
                    'semantics': 'batch',
                  },
                  {
                    'size': 256,
                    'alignment': 0,
                    'semantics': 'spatial',
                  },
                ],
              },
            },
            'children': [],
          },
          {
            'name': '%fusion',
            'metrics': {
              'time': 0.4,
              'flops': 0.22,
            },
            'xla': {
              'op': 'fusion:kOutput',
              'expression': '%fusion = something',
              'provenance': '',
              'category': 'Good ops',
            },
            'children': [
              {
                'name': '%dot.1',
                'xla': {
                  'op': 'dot',
                  'expression': '%dot.1 = something',
                  'provenance': 'TfOpForFusedDot',
                  'category': 'Good ops',
                },
                'children': [],
              },
              {
                'name': '%dot.2',
                'xla': {
                  'op': 'dot',
                  'expression': '%dot.2 = something',
                  'provenance': '',
                  'category': 'Good ops',
                },
                'children': [],
              },
            ],
          },
        ],
      },
      {
        'name': 'Overhead',
        'metrics': {
          'time': 0.4,
          'flops': 0,
        },
        'category': {},
        'children': [
          {
            'name': '%infeed',
            'metrics': {
              'time': 0.4,
              'flops': 0,
            },
            'xla': {
              'op': 'infeed',
              'expression': '%infeed = something',
              'provenance': '',
              'category': 'Overhead',
            },
            'children': [],
          },
        ],
      },
    ],
  },
  'byProgram': {
    'name': 'byProgram',
    'metrics': {
      'time': 1,
      'flops': 0.37,
      'memoryBandwidth': 0.057,
    },
    'children': [],
  },
};

/** Mock data for data API with pod_viewer tag */
export const DATA_PLUGIN_PROFILE_POD_VIEWER_DATA = {
  'podStatsSequence': {
    'podStatsMap': [
      {
        'allReduceOpDb': [
          {
            'dataSize': '30661632',
            'durationUs': 10811.50,
            'name': 'all-reduce',
            'replicaGroups': [
              {'replicaIds': ['0']},
            ]
          },
          {
            'dataSize': '2048',
            'durationUs': 2521.16,
            'name': 'all-reduce.1',
            'replicaGroups': [
              {'replicaIds': ['0']},
            ]
          },
          {
            'dataSize': '12356608',
            'durationUs': 463.57,
            'name': 'all-reduce.2',
            'replicaGroups': [
              {'replicaIds': ['0']},
            ]
          },
        ],
        'channelDb': [
          {
            'channelId': '16436',
            'dataSize': '2764800',
            'durationUs': 34.01,
            'hloNames': [
              'send.42',
              'recv-done.52',
            ],
            'srcCoreIds': [0, 3],
            'dstCoreIds': [1, 2],
            'sendDelayUs': 0,
          },
          {
            'channelId': '17643',
            'dataSize': '20480',
            'durationUs': 276.6,
            'hloNames': [
              'recv-done.100',
              'send.83',
            ],
            'srcCoreIds': [3],
            'dstCoreIds': [0],
            'sendDelayUs': 73.25,
          },
        ],
        'coreIdToReplicaIdMap': {
          '0': 0,
          '1': 0,
          '2': 0,
          '3': 0,
        },
        'podStatsPerCore': {
          '0': {
            'chipId': 1,
            'allReduceComputeDurationUs': 3296.22,
            'allReduceSyncDurationUs': 0.40571,
            'highFlopsComputeUs': 207233.68,
            'hostInfeedDurationUs': 1714.96,
            'hostName': 'njsw1:14059',
            'hostOutfeedDurationUs': 0,
            'nodeId': 0,
            'recvDurationUs': 40773.72,
            'sendDurationUs': 18154.35,
            'totalDurationUs': 398215.85,
          },
          '1': {
            'chipId': 1,
            'allReduceComputeDurationUs': 3296.22,
            'allReduceSyncDurationUs': 0.40571,
            'highFlopsComputeUs': 207201.06,
            'hostInfeedDurationUs': 1719.77,
            'hostName': 'njsw1:14059',
            'hostOutfeedDurationUs': 0,
            'nodeId': 1,
            'recvDurationUs': 37772.05,
            'sendDurationUs': 17354.91,
            'totalDurationUs': 398223.51,
          },
          '2': {
            'chipId': 0,
            'allReduceComputeDurationUs': 3296.22,
            'allReduceSyncDurationUs': 36.02,
            'highFlopsComputeUs': 214142.01,
            'hostInfeedDurationUs': 1725.31,
            'hostName': 'njsw1:14059',
            'hostOutfeedDurationUs': 0.33,
            'nodeId': 0,
            'recvDurationUs': 31277.58,
            'sendDurationUs': 9243.76,
            'totalDurationUs': 398218.81,
          },
          '3': {
            'chipId': 0,
            'allReduceComputeDurationUs': 3296.22,
            'allReduceSyncDurationUs': 36.02,
            'highFlopsComputeUs': 209330.51,
            'hostInfeedDurationUs': 1720.71,
            'hostName': 'njsw1:14059',
            'hostOutfeedDurationUs': 0,
            'nodeId': 1,
            'recvDurationUs': 22356.51,
            'sendDurationUs': 15068.79,
            'totalDurationUs': 398221.08,
          },
        },
        'stepNum': 222,
      },
    ]
  },
  'runEnvironment': {
    'topology': {
      'xDimension': '2',
      'yDimension': '1',
    },
    'tpuType': 'TPU v2',
  },
};

/** Mock data for data API with kernel_stats tag */
export const DATA_PLUGIN_PROFILE_KERNEL_STATS_DATA = [
  {
    'cols': [
      {'id': 'rank', 'label': 'Rank', 'type': 'number'},
      {'id': 'kernel_name', 'label': 'Kernel Name', 'type': 'string'},
      {
        'id': 'registers_per_thread',
        'label': 'Registers per thread',
        'type': 'number'
      },
      {'id': 'shmem_bytes', 'label': 'Shared Mem bytes', 'type': 'number'},
      {'id': 'block_dim', 'label': 'Block dim', 'type': 'string'},
      {'id': 'grid_dim', 'label': 'Grid dim', 'type': 'string'},
      {
        'id': 'is_op_tensor_core_eligible',
        'label': 'Op is TensorCore eligible',
        'type': 'boolean'
      },
      {
        'id': 'is_kernel_using_tensor_core',
        'label': 'Kernel uses TensorCore',
        'type': 'boolean'
      },
      {'id': 'op_name', 'label': 'Op Name', 'type': 'string'},
      {'id': 'occurrences', 'label': 'Occurrences', 'type': 'number'},
      {
        'id': 'total_duration_us',
        'label': 'Total Duration (us)',
        'type': 'number'
      },
      {'id': 'avg_duration_us', 'label': 'Avg Duration (us)', 'type': 'number'},
      {'id': 'min_duration_us', 'label': 'Min Duration (us)', 'type': 'number'},
      {'id': 'max_duration_us', 'label': 'Max Duration (us)', 'type': 'number'},
    ],
    'rows': [
      {
        'c': [
          {'v': 1}, {'v': 'volta_sgemm_64x64_nn'}, {'v': 126}, {'v': 8448},
          {'v': '64,1,1'}, {'v': '4,1547,1'}, {'v': true}, {'v': false},
          {'v': 'model_1/dense/MatMul'}, {'v': 520}, {'v': 473852.691},
          {'v': 911.261}, {'v': 904.191}, {'v': 922.621}
        ]
      },
      {
        'c': [
          {'v': 2}, {'v': 'volta_sgemm_32x128_nn'}, {'v': 55}, {'v': 16384},
          {'v': '256,1,1'}, {'v': '4,774,1'}, {'v': false}, {'v': false},
          {'v': 'model_1/dense_1/MatMul'}, {'v': 520}, {'v': 265960.752},
          {'v': 511.462}, {'v': 505.852}, {'v': 519.17}
        ]
      },
      {
        'c': [
          {'v': 3}, {
            'v': 'void tensorflow::BiasNHWCKernel(int, float const*, float co' +
                'nst*, flot*, int)'
          },
          {'v': 20}, {'v': 0}, {'v': '1024,1,1'}, {'v': '80,1,1'}, {'v': false},
          {'v': true}, {'v': 'model_1/dense/BiasAdd'}, {'v': 520},
          {'v': 153061.981}, {'v': 294.352}, {'v': 289.791}, {'v': 302.082}
        ]
      },
      {
        'c': [
          {'v': 4}, {
            'v': 'void Eigen::internal::EigenMetaKernel const, Eigen::array c' +
                'onst, Eigen::TensorMap, 16, Eigen::MakePointer> >, Eigen::Te' +
                'nsorMap, 16, Eigen::MakePointer> const> const, Eigen::GpuDev' +
                'ice>, int>(Eigen::TensorEvaluator const, Eigen::array const,' +
                ' Eigen::TensorMap, 16, Eigen::MakePointer> >, Eigen::TensorM' +
                'ap, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>,' +
                ' int)'
          },
          {'v': 16}, {'v': 0}, {'v': '1024,1,1'}, {'v': '160,1,1'}, {'v': true},
          {'v': true}, {'v': 'model_1/concatenate/concat'}, {'v': 1040},
          {'v': 142534.251}, {'v': 137.052}, {'v': 134.142}, {'v': 141.313}
        ]
      },
      {
        'c': [
          {'v': 5}, {
            'v': 'void Eigen::internal::EigenMetaKernel, 16, Eigen::MakePoint' +
                'er>, Eigen::TensorCwiseBinaryOp, Eigen::TensorMap, 16, Eigen' +
                '::MakePointer> const, Eigen::TensorCwiseBinaryNullaryOp, Eig' +
                'en::TensorMap, 16, Eigen::MakePointer> const> const> const> ' +
                'const, Eigen::GpuDevice>, long>(Eigne::TensorEvaluator, 16, ' +
                'Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp, Eigen::Tens' +
                'orMap, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBina' +
                'ryNullaryOp, Eigen::TensorMap, 16, Eigen::MakePointer> const' +
                '> const> const> const, Eigen::GpuDevice>, long)'
          },
          {'v': 26}, {'v': 0}, {'v': '1024,1,1'}, {'v': '160,1,1'},
          {'v': false}, {'v': false}, {'v': 'model_1/dense/Relu'}, {'v': 520},
          {'v': 132006.561}, {'v': 253.861}, {'v': 250.881}, {'v': 258.051}
        ]
      },
      {
        'c': [
          {'v': 6}, {
            'v': 'void tensorflow::GatherOpKernel(float const*, int const*, f' +
                'loat*, long long, long long, long long, long long)'
          },
          {'v': 22}, {'v': 0}, {'v': '1024,1,1'}, {'v': '80,1,1'}, {'v': true},
          {'v': true}, {'v': 'model_1/embedding_user/embedding_lookup'},
          {'v': 520}, {'v': 128865.962}, {'v': 247.821}, {'v': 244.742},
          {'v': 253.95}
        ]
      },
      {
        'c': [
          {'v': 7}, {
            'v': 'void tensorflow::GatherOpKernel(float const*, int const*, f' +
                'loat*, long long, long long, long long, long long)'
          },
          {'v': 22}, {'v': 0}, {'v': '1024,1,1'}, {'v': '80,1,1'}, {'v': true},
          {'v': true}, {'v': 'model_1/embedding_item/embedding_lookup'},
          {'v': 520}, {'v': 124522.141}, {'v': 239.461}, {'v': 236.542},
          {'v': 247.811}
        ]
      },
      {
        'c': [
          {'v': 8}, {
            'v': 'void tensorflow::BiasNHWCKernel(int, float const*, float co' +
                'nst*, float*, int)'
          },
          {'v': 20}, {'v': 0}, {'v': '1024,1,1'}, {'v': '80,1,1'}, {'v': true},
          {'v': false}, {'v': 'model_1/dense_1/BiasAdd'}, {'v': 520},
          {'v': 76828.481}, {'v': 147.751}, {'v': 145.411}, {'v': 155.652}
        ]
      },
      {
        'c': [
          {'v': 9}, {'v': 'volta_sgemm_32x128_nn'}, {'v': 55}, {'v': 16384},
          {'v': '256,1,1'}, {'v': '2,774,1'}, {'v': false}, {'v': false},
          {'v': 'model_1/dense_2/MatMul'}, {'v': 519}, {'v': 76154.662},
          {'v': 146.731}, {'v': 144.382}, {'v': 148.481}
        ]
      },
      {
        'c': [
          {'v': 10}, {
            'v': 'void Eigen::internal::EigenMetaKernel const, Eigen::array c' +
                'onst, Eigen::TensorMap, 16, Eigen::MakePointer> >, Eigen::Te' +
                'nsorMap, 16, Eigen::MakePointer> const> const, Eigen::GpuDev' +
                'ice>, int>(Eigen::TensorEvaluator const, Eigen::array const,' +
                ' Eigen::TensorMap, 16, Eigen::MakePointer> >, Eigen::TensorM' +
                'ap, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>,' +
                ' int)'
          },
          {'v': 16}, {'v': 0}, {'v': '1024,1,1'}, {'v': '160,1,1'},
          {'v': false}, {'v': true}, {'v': 'model_1/concatenate_1/concat'},
          {'v': 1038}, {'v': 73137.971}, {'v': 70.461}, {'v': 68.611},
          {'v': 72.702}
        ]
      },
      {
        'c': [
          {'v': 11}, {
            'v': 'void Eigen::internal::EigenMetaKernel, 16, Eigen::MakePoint' +
                'er>, Eigen::TensorCwiseBinaryOp, Eigen::TensorMap, 16, Eigen' +
                '::MakePointer> const, Eigen::TensorCwiseBinaryNullaryOp, Eig' +
                'en::TensorMap, 16, Eigen::MakePointer> const> const> const> ' +
                'const, Eigen::GpuDevice>, int>(Eigne::TensorEvaluator, 16, E' +
                'igen::MakePointer>, Eigen::TensorCwiseBinaryOp, Eigen::Tenso' +
                'rMap, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinar' +
                'yNullaryOp, Eigen::TensorMap, 16, Eigen::MakePointer> const>' +
                ' const> const> const, Eigen::GpuDevice>, int)'
          },
          {'v': 16}, {'v': 0}, {'v': '1024,1,1'}, {'v': '160,1,1'},
          {'v': false}, {'v': false},
          {'v': 'model_1/embedding_user_mip/strided_slice'}, {'v': 520},
          {'v': 70798.131}, {'v': 136.151}, {'v': 133.121}, {'v': 148.482}
        ]
      },
      {
        'c': [
          {'v': 12}, {
            'v': 'void Eigen::internal::EigenMetaKernel, 16, Eigen::MakePoint' +
                'er>, Eigen::TensorCwiseBinaryOp, Eigen::TensorMap, 16, Eigen' +
                '::MakePointer> const, Eigen::TensorCwiseBinaryNullaryOp, Eig' +
                'en::TensorMap, 16, Eigen::MakePointer> const> const> const> ' +
                'const, Eigen::GpuDevice>, int>(Eigne::TensorEvaluator, 16, E' +
                'igen::MakePointer>, Eigen::TensorCwiseBinaryOp, Eigen::Tenso' +
                'rMap, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinar' +
                'yNullaryOp, Eigen::TensorMap, 16, Eigen::MakePointer> const>' +
                ' const> const> const, Eigen::GpuDevice>, int)'
          },
          {'v': 16}, {'v': 0}, {'v': '1024,1,1'}, {'v': '160,1,1'},
          {'v': false}, {'v': false},
          {'v': 'model_1/embedding_item_mip/strided_slice'}, {'v': 520},
          {'v': 70783.822}, {'v': 136.121}, {'v': 133.121}, {'v': 149.502}
        ]
      },
    ],
    'p': {},
  },
];
