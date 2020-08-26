import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {SimpleDataTableOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

declare interface KernelStatsColumn {
  kernelName: number;
  totalDurationUs: number;
}

const KERNEL_NAME_COLUMN_INDEX = 0;
const TOTAL_DURATION_US_COLUMN_INDEX = 1;
const TOOLTIP_COLUMN_INDEX = 2;

/** A kernel stats chart view component. */
@Component({
  selector: 'kernel-stats-chart',
  templateUrl: './kernel_stats_chart.ng.html',
  styleUrls: ['./kernel_stats_chart.scss']
})
export class KernelStatsChart implements OnChanges, OnInit {
  /** The kernel stats data. */
  @Input() kernelStatsData: SimpleDataTableOrNull = null;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  chart?: google.visualization.PieChart;
  dataTable?: google.visualization.DataTable;
  count = 0;
  total = 0;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  private makeTooltip(
      kernelName: string, durationUs: number, totalDurationUs: number): string {
    return '<div style="padding:5px;">' +
        '<b>' + kernelName + '</b><br/>' + durationUs.toFixed(2) + ' us ' +
        '(' + (durationUs / totalDurationUs * 100.0).toFixed(1) + ')%' +
        '</div>';
  }

  private getDataTable(): google.visualization.DataTable|null {
    if (!this.kernelStatsData) {
      return null;
    }

    if (this.dataTable) {
      return this.dataTable;
    }

    const dataTable = new google.visualization.DataTable(this.kernelStatsData);
    const columns: KernelStatsColumn = {
      kernelName: 0,
      totalDurationUs: 0,
    };
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      switch (dataTable.getColumnId(i)) {
        case 'kernel_name':
          columns.kernelName = i;
          break;
        case 'total_duration_us':
          columns.totalDurationUs = i;
          break;
        default:
          break;
      }
    }

    // tslint:disable-next-line:no-any
    const dataGroup = new (google.visualization as any)['data']['group'](
        dataTable, [columns.kernelName], [
          {
            'column': columns.totalDurationUs,
            // tslint:disable-next-line:no-any
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
        ]);
    dataGroup.sort([
      {
        column: TOTAL_DURATION_US_COLUMN_INDEX,
        desc: true,
      },
      {
        column: KERNEL_NAME_COLUMN_INDEX,
        asc: true,
      },
    ]);
    dataGroup.addColumn({
      type: 'string',
      role: 'tooltip',
      p: {'html': true},
    });

    this.total = dataGroup.getNumberOfRows();
    this.count = Math.min(this.total, 10);

    this.dataTable = new google.visualization.DataView(dataGroup).toDataTable();

    return this.dataTable;
  }

  drawChart() {
    if (!this.chart) {
      return;
    }

    const dataTable = this.getDataTable();
    if (!dataTable) {
      return;
    }

    let totalDurationUs = 0.0;
    for (let i = 0; i < this.count; i++) {
      totalDurationUs += dataTable.getValue(i, TOTAL_DURATION_US_COLUMN_INDEX);
    }

    for (let i = 0; i < this.count; i++) {
      const tooltip = this.makeTooltip(
          dataTable.getValue(i, KERNEL_NAME_COLUMN_INDEX),
          dataTable.getValue(i, TOTAL_DURATION_US_COLUMN_INDEX),
          totalDurationUs);
      dataTable.setCell(i, TOOLTIP_COLUMN_INDEX, tooltip);
    }

    const dataView = new google.visualization.DataView(dataTable);
    if (this.total > this.count) {
      dataView.hideRows(this.count, this.total - 1);
    }

    const options = {
      backgroundColor: 'transparent',
      width: 700,
      height: 250,
      chartArea: {
        left: 0,
        width: '90%',
        height: '90%',
      },
      legend: {textStyle: {fontSize: 10}},
      sliceVisibilityThreshold: 0.01,
      tooltip: {isHtml: true},
    };

    this.chart.draw(dataView, options);
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
          new google.visualization.PieChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }

  updateCount(value: number) {
    value = Number(value);
    if (value === this.count) {
      return;
    }

    this.count = value;
    this.drawChart();
  }
}
