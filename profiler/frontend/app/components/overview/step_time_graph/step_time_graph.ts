import {AfterViewInit, Component, ElementRef, HostListener, Input, OnChanges, SimpleChanges, ViewChild} from '@angular/core';

import {InputPipelineAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';

const MAX_CHART_WIDTH = 800;
const COLUMN_IDS_FOR_TPU = [
  'stepnum',
  'computeTimeMs',
  'inputTimeMs',
  'idleTimeMs',
  'tooltip',
];
const COLUMN_IDS_FOR_GPU = [
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

/** A step-time graph view component. */
@Component({
  selector: 'step-time-graph',
  templateUrl: './step_time_graph.ng.html',
  styleUrls: ['./step_time_graph.scss']
})
export class StepTimeGraph implements AfterViewInit, OnChanges {
  /** The input pipeline analyis data. */
  @Input() inputPipelineAnalysis: InputPipelineAnalysis|null = null;

  /** The default column ids. */
  @Input() columnIds = COLUMN_IDS_FOR_TPU;

  /** The default column colors. */
  @Input() columnColors = COLORS_FOR_TPU;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  title = 'Step-time Graph';
  height = 300;
  width = 0;
  chart: google.visualization.AreaChart|null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.width = 0;
    this.drawChart();
  }

  @HostListener('window:resize')
  onResize() {
    this.drawChart();
  }

  drawChart() {
    if (!this.chartRef) {
      return;
    }

    const newWidth =
        Math.min(MAX_CHART_WIDTH, this.chartRef.nativeElement.offsetWidth);

    if (!this.chart || !this.inputPipelineAnalysis || this.width === newWidth) {
      return;
    }
    this.width = newWidth;

    const dataTable =
        new google.visualization.DataTable(this.inputPipelineAnalysis);
    let columnsIds = this.columnIds;
    let colors = this.columnColors;
    this.height = 300;
    this.inputPipelineAnalysis.p = this.inputPipelineAnalysis.p || {};
    if ((this.inputPipelineAnalysis.p.hardware_type || 'TPU') !== 'TPU') {
      columnsIds = COLUMN_IDS_FOR_GPU;
      colors = COLORS_FOR_GPU;
      this.height = 400;
    }

    let i = 0;
    while (i < dataTable.getNumberOfColumns()) {
      if (!columnsIds.includes(dataTable.getColumnId(i))) {
        dataTable.removeColumn(i);
        continue;
      }
      i++;
    }

    const showTextEvery =
        Math.max(1, Math.floor(dataTable.getNumberOfRows() / 10));
    const options = {
      title: 'Step Time (in milliseconds)',
      titleTextStyle: {bold: true},
      hAxis: {
        title: 'Step Number',
        showTextEvery,
        textStyle: {bold: true},
      },
      vAxis: {
        format: '###.####',
        minValue: 0,
        textStyle: {bold: true},
      },
      chartArea: {left: 50, width: '60%'},
      colors: colors,
      height: this.height,
      isStacked: true,
    };
    this.chart.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      this.chart =
          new google.visualization.AreaChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }
}
