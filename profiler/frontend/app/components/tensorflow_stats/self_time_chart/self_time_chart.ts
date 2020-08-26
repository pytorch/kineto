import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';

import {OpExecutor, OpKind} from 'org_xprof/frontend/app/common/constants/enums';
import {TensorflowStatsDataOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A self-time chart view component. */
@Component({
  selector: 'self-time-chart',
  templateUrl: './self_time_chart.ng.html',
  styleUrls: ['./self_time_chart.scss']
})
export class SelfTimeChart implements OnChanges, OnInit {
  /** The tensorflow stats data. */
  @Input() tensorflowStatsData: TensorflowStatsDataOrNull = null;

  /** The Op executor. */
  @Input() opExecutor: OpExecutor = OpExecutor.NONE;

  /** The Op kind. */
  @Input() opKind: OpKind = OpKind.NONE;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  title = '';
  chart: google.visualization.PieChart|null = null;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.chart || !this.tensorflowStatsData || !this.opExecutor ||
        !this.opKind) {
      return;
    }

    const dataTable =
        new google.visualization.DataTable(this.tensorflowStatsData);
    const numberFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    let dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{
      column: 1,
      value: this.opExecutor,
    }]));

    this.title =
        'ON ' + String(this.opExecutor).toUpperCase() + ': TOTAL SELF-TIME';
    if (this.opKind === OpKind.TYPE) {
      this.title += ' (GROUPED BY TYPE)';
      const dataGroup = new (google.visualization as any)['data']['group'](
          dataView, [2], [{
            'column': 7,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          }]);
      dataGroup.sort({column: 1, desc: true});
      numberFormatter.format(dataGroup, 1);
      dataView = new google.visualization.DataView(dataGroup);
    } else {
      numberFormatter.format(dataTable, 7);
      dataView.setColumns([3, 7]);
    }

    const options = {
      backgroundColor: 'transparent',
      width: 400,
      height: 200,
      chartArea: {
        left: 0,
        width: '100%',
        height: '80%',
      },
      legend: {textStyle: {fontSize: 10}},
      sliceVisibilityThreshold: 0.01,
    };

    this.chart.draw(dataView, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['corechart']})
    google.charts.setOnLoadCallback(() => {
      this.chart =
          new google.visualization.PieChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }
}
