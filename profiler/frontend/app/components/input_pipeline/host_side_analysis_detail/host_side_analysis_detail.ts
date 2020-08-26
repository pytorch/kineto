import {AfterViewInit, Component, ElementRef, Input, OnChanges, SimpleChanges, ViewChild} from '@angular/core';

import {HostOpsColumn, InputPipelineDeviceAnalysisOrNull, InputPipelineHostAnalysisOrNull, SimpleDataTableOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A host-side analysis detail view component. */
@Component({
  selector: 'host-side-analysis-detail',
  templateUrl: './host_side_analysis_detail.ng.html',
  styleUrls: ['./host_side_analysis_detail.scss']
})
export class HostSideAnalysisDetail implements AfterViewInit, OnChanges {
  /** The input pipeline host anaysis data. */
  @Input() hostAnalysis: InputPipelineHostAnalysisOrNull = null;

  /** The recommendation data. */
  @Input()
  set recommendation(data: SimpleDataTableOrNull) {
    data = data || {};
    data.rows = data.rows || [];
    data.rows.forEach(row => {
      if (row.c && row.c[0] && row.c[0].v) {
        this.recommendations.push(String(row.c[0].v));
      }
    });
  }

  @ViewChild('columnChart', {static: false}) columnChartRef!: ElementRef;
  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  hasHostOps = false;
  recommendations: string[] = [];
  columnChart: google.visualization.ColumnChart|null = null;
  table: google.visualization.Table|null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.hostAnalysis || !this.columnChart || !this.table) {
      return;
    }

    this.drawColumnChart();
    this.drawTable();
  }

  drawColumnChart() {
    if (!this.hostAnalysis || !this.columnChart) {
      return;
    }

    const kUsPerMs = 1000.0;
    const p = this.hostAnalysis.p || {};
    const unclassifiedNonEnqueueMs =
        Number(p.unclassified_nonequeue_us) / kUsPerMs;
    const demandedFileReadMs = Number(p.demanded_file_read_us) / kUsPerMs;
    const advancedFileReadMs = Number(p.advanced_file_read_us) / kUsPerMs;
    const preprocessingMs = Number(p.preprocessing_us) / kUsPerMs;
    const enqueueMs = Number(p.enqueue_us) / kUsPerMs;

    const dataTable = google.visualization.arrayToDataTable([
      [
        'Input time breakdown',
        'Other data reading or processing (in ms)',
        'Reading data from files on demand (in ms)',
        'Reading data from files in advance [including caching, prefetching, interleaving] (in ms)',
        'Data preprocessing (in ms)',
        'Enqueuing data to be transferred to device (in ms)',
      ],
      [
        '',
        unclassifiedNonEnqueueMs,
        demandedFileReadMs,
        advancedFileReadMs,
        preprocessingMs,
        enqueueMs,
      ],
    ]);

    const options = {
      bar: {groupWidth: '45%'},
      chartArea: {
        left: 40,
        width: '50%',
        height: '90%',
      },
      width: 800,
      height: 300,
      colors: ['red', 'blue', 'orange', 'green', 'purple'],
      backgroundColor: {'fill': 'transparent'},
      isStacked: 'percent',
    };
    this.columnChart.draw(
        dataTable, options as google.visualization.ColumnChartOptions);
  }

  drawTable() {
    if (!this.hostAnalysis || !this.table) {
      return;
    }

    const dataTable = new google.visualization.DataTable(this.hostAnalysis);
    if (dataTable.getNumberOfRows() < 1) {
      this.hasHostOps = false;
      return;
    }
    this.hasHostOps = true;

    const columns: HostOpsColumn = {
      opName: 0,
      count: 0,
      timeInMs: 0,
      timeInPercent: 0,
      selfTimeInMs: 0,
      selfTimeInPercent: 0,
      category: 0,
    };
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      switch (dataTable.getColumnId(i)) {
        case 'opName':
          columns.opName = i;
          break;
        case 'count':
          columns.count = i;
          break;
        case 'timeInMs':
          columns.timeInMs = i;
          break;
        case 'timeInPercent':
          columns.timeInPercent = i;
          break;
        case 'selfTimeInMs':
          columns.selfTimeInMs = i;
          break;
        case 'selfTimeInPercent':
          columns.selfTimeInPercent = i;
          break;
        case 'category':
          columns.category = i;
          break;
        default:
          break;
      }
    }

    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(dataTable, columns.timeInPercent);
    percentFormatter.format(dataTable, columns.selfTimeInPercent);

    const zeroDecimalPtFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    zeroDecimalPtFormatter.format(dataTable, columns.timeInMs);
    zeroDecimalPtFormatter.format(dataTable, columns.selfTimeInMs);

    dataTable.setProperty(0, columns.opName, 'style', 'width: 40%');
    dataTable.setProperty(0, columns.count, 'style', 'width: 15%');
    dataTable.setProperty(0, columns.timeInMs, 'style', 'width: 10%');
    dataTable.setProperty(0, columns.timeInPercent, 'style', 'width: 5%');
    dataTable.setProperty(0, columns.selfTimeInMs, 'style', 'width: 10%');
    dataTable.setProperty(0, columns.selfTimeInPercent, 'style', 'width: 5%');
    dataTable.setProperty(0, columns.category, 'style', 'width: 15%');
    const options = {
      alternatingRowStyle: false,
      showRowNumber: false,
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      }
    };

    this.table.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['corechart', 'table']})
    google.charts.setOnLoadCallback(() => {
      this.columnChart = new google.visualization.ColumnChart(
          this.columnChartRef.nativeElement);
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawChart();
    });
  }
}
