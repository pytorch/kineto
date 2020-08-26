import {Component, ElementRef, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges, ViewChild} from '@angular/core';

import {TensorflowStatsDataOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A flop rate chart view component. */
@Component({
  selector: 'flop-rate-chart',
  templateUrl: './flop_rate_chart.ng.html',
  styleUrls: ['./flop_rate_chart.scss']
})
export class FlopRateChart implements OnChanges, OnInit {
  /** The tensorflow stats data. */
  @Input() tensorflowStatsData: TensorflowStatsDataOrNull = null;

  /** The event to notify whether the data has rows. */
  @Output() hasDataRowsChanged = new EventEmitter<boolean>();

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  chart: google.visualization.ColumnChart|null = null;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.chart || !this.tensorflowStatsData) {
      return;
    }

    const dataTable =
        new google.visualization.DataTable(this.tensorflowStatsData);
    const numberFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 1});
    numberFormatter.format(dataTable, 13);

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{
      column: 13,
      minValue: 0.0000,
    }]));
    dataView.setColumns([3, 13]);

    this.hasDataRowsChanged.emit(dataView.getNumberOfRows() > 0);

    const options = {
      backgroundColor: 'transparent',
      strokeWidth: 2,
      width: 550,
      height: 200,
      chartArea: {
        left: 70,
        top: 10,
        width: '80%',
        height: '80%',
      },
      hAxis: {
        textPosition: 'none',
        title: 'TensorFlow Op on Device (in decreasing total self-time)',
      },
      vAxis: {title: 'GFLOPs/sec'},
      legend: {position: 'none'},
      tooltip: {isHtml: true, 'ignoreBounds': true},
    };

    this.chart.draw(
        dataView, options as google.visualization.ColumnChartOptions);
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
          new google.visualization.ColumnChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }
}
