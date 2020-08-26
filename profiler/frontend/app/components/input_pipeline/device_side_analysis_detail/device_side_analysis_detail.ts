import {AfterViewInit, Component, ElementRef, Input, OnChanges, SimpleChanges, ViewChild} from '@angular/core';

import {InputPipelineDeviceAnalysisOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

const INFEED_COLUMN_IDS = [
  'stepnum',
  'infeedPercentAverage',
  'infeedPercentMin',
  'infeedPercentMax',
];
const STEPTIME_COLUMN_IDS_FOR_TPU = [
  'stepnum',
  'computeTimeMs',
  'inputTimeMs',
  'idleTimeMs',
  'tooltip',
];
const STEPTIME_COLUMN_IDS_FOR_GPU = [
  'stepnum',
  'deviceComputeTimeMs',
  'deviceToDeviceTimeMs',
  'hostComputeTimeMs',
  'kernelLaunchTimeMs',
  'infeedTimeMs',
  'outfeedTimeMs',
  'compileTimeMs',
  'otherTimeMs',
  'tooltip',
];
const COLORS_FOR_TPU = [
  'green',
  'crimson',
  'blue',
];
const COLORS_FOR_GPU = [
  '#9CCC65',
  '#FFEB3B',
  '#64B5F6',
  '#FF9800',
  '#B71C1C',
  'black',
  '#00695C',
  '#5E35B1',
];

interface DeviceSideAnalysisMetrics {
  average?: string;
  max?: string;
  min?: string;
  stddev?: string;
}

declare interface Intervals {
  style?: string;
  color?: string;
}

declare interface ExtendedLineChartOptions extends
    google.visualization.LineChartOptions {
  intervals?: Intervals;
}

/** A device-side analysis detail view component. */
@Component({
  selector: 'device-side-analysis-detail',
  templateUrl: './device_side_analysis_detail.ng.html',
  styleUrls: ['./device_side_analysis_detail.scss']
})
export class DeviceSideAnalysisDetail implements AfterViewInit, OnChanges {
  /** The input pipeline device analysis data. */
  @Input()
  set deviceAnalysis(analysis: InputPipelineDeviceAnalysisOrNull) {
    this.inputPipelineDeviceAnalysis = analysis;
    analysis = analysis || {};
    analysis.p = analysis.p || {};
    if (!analysis.rows || analysis.rows.length === 0) {
      return;
    }
    this.isTpu = (analysis.p.hardware_type || 'TPU') === 'TPU';
    this.steptimeMsMetrics.average = analysis.p.steptime_ms_average || '';
    this.steptimeMsMetrics.max = analysis.p.steptime_ms_maximum || '';
    this.steptimeMsMetrics.min = analysis.p.steptime_ms_minimum || '';
    this.steptimeMsMetrics.stddev =
        analysis.p.steptime_ms_standard_deviation || '';

    this.infeedPercentMetrics.average = analysis.p.infeed_percent_average || '';
    this.infeedPercentMetrics.max = analysis.p.infeed_percent_maximum || '';
    this.infeedPercentMetrics.min = analysis.p.infeed_percent_minimum || '';
    this.infeedPercentMetrics.stddev =
        analysis.p.infeed_percent_standard_deviation || '';
  }

  /** The default column ids. */
  @Input() columnIds = STEPTIME_COLUMN_IDS_FOR_TPU;

  /** The default column colors. */
  @Input() columnColors = COLORS_FOR_TPU;

  @ViewChild('areaChart', {static: false}) areaChartRef!: ElementRef;
  @ViewChild('lineChart', {static: false}) lineChartRef!: ElementRef;

  isTpu = true;
  inputPipelineDeviceAnalysis: InputPipelineDeviceAnalysisOrNull = null;
  infeedPercentMetrics: DeviceSideAnalysisMetrics = {};
  steptimeMsMetrics: DeviceSideAnalysisMetrics = {};
  areaChart: google.visualization.AreaChart|null = null;
  lineChart: google.visualization.LineChart|null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.inputPipelineDeviceAnalysis || !this.lineChart ||
        !this.areaChart) {
      return;
    }

    const dataTable =
        new google.visualization.DataTable(this.inputPipelineDeviceAnalysis);
    this.drawAreaChart(dataTable.clone());
    this.drawLineChart(dataTable.clone());
  }

  drawLineChart(dataTable: google.visualization.DataTable|null) {
    if (!this.lineChart || !dataTable) {
      return;
    }

    let i = 0;
    while (i < dataTable.getNumberOfColumns()) {
      if (!INFEED_COLUMN_IDS.includes(dataTable.getColumnId(i))) {
        dataTable.removeColumn(i);
        continue;
      }
      i++;
    }

    const options: ExtendedLineChartOptions = {
      hAxis: {title: 'Step Number'},
      vAxis: {title: '% of step time', format: '###.###\'%\''},
      chartArea: {
        left: 100,
        top: 10,
        width: '65%',
        height: '90%',
      },
      width: 820,
      height: 300,
      legend: 'none',
      lineWidth: 1,
      colors: ['none'],
      backgroundColor: {fill: 'transparent'},
      intervals: {style: 'boxes', color: 'red'},
    };
    this.lineChart.draw(dataTable, options);
  }

  drawAreaChart(dataTable: google.visualization.DataTable|null) {
    if (!this.areaChart || !dataTable) {
      return;
    }

    let i = 0;
    let columnsIds = this.columnIds;
    let colors = this.columnColors;
    if (!this.isTpu) {
      columnsIds = STEPTIME_COLUMN_IDS_FOR_GPU;
      colors = COLORS_FOR_GPU;
    }
    while (i < dataTable.getNumberOfColumns()) {
      if (!columnsIds.includes(dataTable.getColumnId(i))) {
        dataTable.removeColumn(i);
        continue;
      }
      i++;
    }

    const options = {
      hAxis: {title: 'Step Number'},
      vAxis: {title: 'Milliseconds', format: '###.####ms', minValue: 0},
      chartArea: {
        left: 100,
        top: 10,
        width: '65%',
        height: '90%',
      },
      width: 820,
      height: 300,
      colors: colors,
      backgroundColor: {fill: 'transparent'},
      isStacked: true,
    };
    this.areaChart.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['corechart']})
    google.charts.setOnLoadCallback(() => {
      this.areaChart =
          new google.visualization.AreaChart(this.areaChartRef.nativeElement);
      this.lineChart =
          new google.visualization.LineChart(this.lineChartRef.nativeElement);
      this.drawChart();
    });
  }
}
