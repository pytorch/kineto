import {Component, ElementRef, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges, ViewChild} from '@angular/core';
import {PrimitiveTypeNumberStringOrUndefined} from 'org_xprof/frontend/app/common/interfaces/data_table';

const BAR_WIDTH = 50;
const DEFAULT_CHART_WIDTH = 500;

/** A stack bar chart view component. */
@Component({
  selector: 'stack-bar-chart',
  templateUrl: './stack_bar_chart.ng.html',
  styleUrls: ['./stack_bar_chart.scss']
})
export class StackBarChart implements OnChanges, OnInit {
  /** The data to be display. */
  @Input() data?: Array<Array<PrimitiveTypeNumberStringOrUndefined>>;

  /** The event when the selection of the chart is changed. */
  @Output() selected = new EventEmitter<number>();

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  chart: google.visualization.BarChart|null = null;
  chartWidth = DEFAULT_CHART_WIDTH;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.chart || !this.data) {
      return;
    }

    this.chartWidth =
        Math.max(DEFAULT_CHART_WIDTH, this.data.length * BAR_WIDTH);
    const dataTable = window.google.visualization.arrayToDataTable(this.data);

    const options = {
      backgroundColor: 'transparent',
      chartArea: {
        left: 50,
        right: 20,
        top: 50,
        bottom: 20,
      },
      focusTarget: 'category',
      isStacked: true,
      legend: {
        position: 'top',
        maxLines: 3,
        textStyle: {fontSize: 12},
      },
      hAxis: {textStyle: {fontSize: 12}},
      vAxis: {textStyle: {fontSize: 12}},
      orientation: 'horizontal',
      tooltip: {trigger: 'none'},
      width: this.chartWidth,
    };

    this.chart.draw(dataTable, options as google.visualization.BarChartOptions);
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
          new google.visualization.BarChart(this.chartRef.nativeElement);

      google.visualization.events.addListener(
          this.chart, 'onmouseover',
          (event: google.visualization.VisualizationSelectionArray) => {
            event = event || {};
            this.selected.emit(event.row || 0);
          });

      this.drawChart();
    });
  }
}
