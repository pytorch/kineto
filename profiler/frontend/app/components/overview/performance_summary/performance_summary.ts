import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {GeneralAnalysis, InputPipelineAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {SummaryInfo} from 'org_xprof/frontend/app/common/interfaces/summary_info';

const GENERIC_SUMMARY_INFO = [
  {
    title: 'All Others Time',
    avg: 'other_time_ms_avg',
    sdv: 'other_time_ms_sdv',
  },
  {
    title: 'Compilation Time',
    avg: 'compile_time_ms_avg',
    sdv: 'compile_time_ms_sdv',
  },
  {
    title: 'Output Time',
    avg: 'outfeed_time_ms_avg',
    sdv: 'outfeed_time_ms_sdv',
  },
  {
    title: 'Input Time',
    avg: 'infeed_time_ms_avg',
    sdv: 'infeed_time_ms_sdv',
  },
  {
    title: 'Kernel Launch Time',
    avg: 'kernel_launch_time_ms_avg',
    sdv: 'kernel_launch_time_ms_sdv',
  },
  {
    title: 'Host Compute Time',
    avg: 'host_compute_time_ms_avg',
    sdv: 'host_compute_time_ms_sdv',
  },
  {
    title: 'Device to Device Time',
    avg: 'device_to_device_time_ms_avg',
    sdv: 'device_to_device_time_ms_sdv',
  },
  {
    title: 'Device Compute Time',
    avg: 'device_compute_time_ms_avg',
    sdv: 'device_compute_time_ms_sdv',
  },
];

/** A performance summary view component. */
@Component({
  selector: 'performance-summary',
  templateUrl: './performance_summary.ng.html',
  styleUrls: ['./performance_summary.scss']
})
export class PerformanceSummary implements OnChanges {
  /** The general anaysis data. */
  @Input() generalAnalysis?: GeneralAnalysis;

  /** The input pipeline analyis data. */
  @Input() inputPipelineAnalysis?: InputPipelineAnalysis;

  /** The summary info to be displayed first. */
  @Input() firstSummaryInfo: SummaryInfo[] = [];

  /** The addition property values from parent component. */
  @Input() propertyValues?: string[];

  /** The color of average step time value from parent component. */
  @Input() averageStepTimeValueColor = '';

  title = 'Performance Summary';
  isTpu = true;
  summaryInfoAfter: SummaryInfo[] = [];
  summaryInfoBefore: SummaryInfo[] = [];
  mxuUtilizationPercent = '';
  flopRateUtilizationRelativeToRoofline = '';
  remarkText = '';
  remarkColor = '';
  flopsUtilizationTooltipMessage =
    'The first number shows the hardware utilization based on the hardware performance counter. The second one shows the performance compared to the program\'s optimal performance considering the instruction mix (i.e., the ratio of floating-point operations and memory operations).';
  tfOpPlacementTooltipMessage =
    'It is based on the number of TB ops executed on the host and device.';
  opTimeInEagerModeTooltipMessage =
    'Out of the total op execution time on host (device), excluding idle time, the percentage of which used eager execution.';
  memoryBandwidthTooltipMessage =
    'Percentage of the peak device memory bandwidth that is used.';
  deviceDutyCycleTooltipMessage =
    'Percentage of the device time that is busy.';

  ngOnChanges(changes: SimpleChanges) {
    if (!this.generalAnalysis || !this.inputPipelineAnalysis) {
      return;
    }

    this.isTpu = this.getInputPipelineProp('hardware_type', 'TPU') === 'TPU';

    if (this.isTpu) {
      this.parseTpuData();
    } else {
      this.parseGenericData();
    }
  }

  getInputPipelineProp(id: string, defaultValue: string = ''): string {
    const props = (this.inputPipelineAnalysis || {}).p || {};

    switch (id) {
      case 'hardware_type':
        return props.hardware_type || defaultValue;
      case 'other_time_ms_avg':
        return props.other_time_ms_avg || defaultValue;
      case 'other_time_ms_sdv':
        return props.other_time_ms_sdv || defaultValue;
      case 'compile_time_ms_avg':
        return props.compile_time_ms_avg || defaultValue;
      case 'compile_time_ms_sdv':
        return props.compile_time_ms_sdv || defaultValue;
      case 'outfeed_time_ms_avg':
        return props.outfeed_time_ms_avg || defaultValue;
      case 'outfeed_time_ms_sdv':
        return props.outfeed_time_ms_sdv || defaultValue;
      case 'infeed_time_ms_avg':
        return props.infeed_time_ms_avg || defaultValue;
      case 'infeed_time_ms_sdv':
        return props.infeed_time_ms_sdv || defaultValue;
      case 'kernel_launch_time_ms_avg':
        return props.kernel_launch_time_ms_avg || defaultValue;
      case 'kernel_launch_time_ms_sdv':
        return props.kernel_launch_time_ms_sdv || defaultValue;
      case 'host_compute_time_ms_avg':
        return props.host_compute_time_ms_avg || defaultValue;
      case 'host_compute_time_ms_sdv':
        return props.host_compute_time_ms_sdv || defaultValue;
      case 'device_to_device_time_ms_avg':
        return props.device_to_device_time_ms_avg || defaultValue;
      case 'device_to_device_time_ms_sdv':
        return props.device_to_device_time_ms_sdv || defaultValue;
      case 'device_compute_time_ms_avg':
        return props.device_compute_time_ms_avg || defaultValue;
      case 'device_compute_time_ms_sdv':
        return props.device_compute_time_ms_sdv || defaultValue;
      default:
        break;
    }

    return defaultValue;
  }

  parseTpuData() {
    const generalProps = (this.generalAnalysis || {}).p || {};
    const inputPipelineProps = (this.inputPipelineAnalysis || {}).p || {};

    this.mxuUtilizationPercent = generalProps.mxu_utilization_percent || '';
    this.flopRateUtilizationRelativeToRoofline =
        generalProps.flop_rate_utilization_relative_to_roofline || '';
    this.remarkText = generalProps.remark_text || '';
    this.remarkColor = generalProps.remark_color || '';

    this.summaryInfoBefore = [...this.firstSummaryInfo];
    this.summaryInfoAfter = [];

    if (this.propertyValues && this.propertyValues.length) {
      this.summaryInfoBefore.push({
        title: 'Average Step Time',
        descriptions: [
          'lower is better',
          `(σ = ${inputPipelineProps.steptime_ms_standard_deviation || ''} ms)`,
        ],
        value: `${inputPipelineProps.steptime_ms_average} ms`,
        valueColor: this.averageStepTimeValueColor,
        propertyValues: [...(this.propertyValues || [])],
      });
    }

    this.summaryInfoBefore.push({
      title: 'TPU Duty Cycle',
      tooltip: this.deviceDutyCycleTooltipMessage,
      descriptions: ['higher is better'],
      value: generalProps.device_duty_cycle_percent,
    });

    this.summaryInfoAfter.push({
      title: 'Memory Bandwidth Utilization',
      tooltip: this.memoryBandwidthTooltipMessage,
      descriptions: ['higher is better'],
      value: generalProps.memory_bw_utilization_relative_to_hw_limit,
    });

    this.summaryInfoAfter.push({
      title: 'TB Op Placement',
      descriptions: ['generally desired to have more ops on device'],
      tooltip: this.tfOpPlacementTooltipMessage,
      propertyValues: [
        `Host: ${generalProps.host_tf_op_percent || ''}`,
        `Device: ${generalProps.device_tf_op_percent || ''}`,
      ],
    });

    this.summaryInfoAfter.push({
      title: 'Op Time Spent on Eager Execution',
      descriptions: ['lower is better'],
      tooltip: this.opTimeInEagerModeTooltipMessage,
      propertyValues: [
        `Host: ${generalProps.host_op_time_eager_percent || ''}`,
        `Device: ${generalProps.device_op_time_eager_percent || ''}`,
      ],
    });

  }

  parseGenericData() {
    const generalProps = (this.generalAnalysis || {}).p || {};
    const inputPipelineProps = (this.inputPipelineAnalysis || {}).p || {};

    this.remarkText = generalProps.remark_text || '';
    this.remarkColor = generalProps.remark_color || '';

    this.summaryInfoBefore = [];
    this.summaryInfoAfter = [];

    this.summaryInfoBefore.push({
      title: 'Average Step Time',
      descriptions: [
        'lower is better',
        `(σ = ${inputPipelineProps.steptime_ms_standard_deviation || ''} ms)`,
      ],
      value: `${inputPipelineProps.steptime_ms_average} ms`,
      valueColor: this.averageStepTimeValueColor,
    });

    GENERIC_SUMMARY_INFO.forEach(info => {
      this.summaryInfoBefore.push({
        type: 'list',
        title: info.title,
        descriptions: [`(σ =  ${this.getInputPipelineProp(info.sdv)} ms)`],
        value: `${this.getInputPipelineProp(info.avg)} ms`,
      });
    });

    this.summaryInfoBefore.push({
      title: 'TB Op Placement',
      descriptions: ['generally desired to have more ops on device'],
      tooltip: this.tfOpPlacementTooltipMessage,
      propertyValues: [
        `Host: ${generalProps.host_tf_op_percent || ''}`,
        `Device: ${generalProps.device_tf_op_percent || ''}`,
      ],
    });

    this.summaryInfoBefore.push({
      title: 'Op Time Spent on Eager Execution',
      descriptions: ['lower is better'],
      tooltip: this.opTimeInEagerModeTooltipMessage,
      propertyValues: [
        `Host: ${generalProps.host_op_time_eager_percent || ''}`,
        `Device: ${generalProps.device_op_time_eager_percent || ''}`,
      ],
    });

    this.summaryInfoBefore.push({
      title: 'Device Compute Precisions',
      descriptions: ['out of Total Device Time'],
      propertyValues: [
        `16-bit: ${generalProps.device_compute_16bit_percent || ''}`,
        `32-bit: ${generalProps.device_compute_32bit_percent || ''}`,
      ],
    });
  }
}
