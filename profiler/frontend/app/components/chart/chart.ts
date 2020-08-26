import {Component, ElementRef, EventEmitter, Input, NgModule, OnChanges, OnInit, Output, SimpleChanges} from '@angular/core';
import {ChartClass, ChartDataInfo, ChartType} from 'org_xprof/frontend/app/common/interfaces/chart';

/** A common chart component. */
@Component({
  selector: 'chart',
  template: '',
  styles: [':host {display: block;}'],
})
export class Chart implements OnChanges, OnInit {
  /** The type of chart. */
  @Input() chartType?: ChartType;

  /** The information of chart data. */
  @Input() dataInfo?: ChartDataInfo;

  /** The event for the number of rows of processed data. */
  @Output() processedNumberOfRows = new EventEmitter<number>();

  chart?: ChartClass;

  constructor(private readonly elementRef: ElementRef) {}

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['dataInfo'] && this.dataInfo && this.dataInfo.dataProvider) {
      this.dataInfo.dataProvider.setData(this.dataInfo);
      this.dataInfo.dataProvider.setSortColumns(
          this.dataInfo.sortColumns || []);
      this.dataInfo.dataProvider.setFilters(this.dataInfo.filters || []);
    }
    this.draw();
  }

  draw() {
    if (!this.chart) {
      return;
    }

    if (!this.dataInfo || !this.dataInfo.dataProvider) {
      this.chart.clearChart();
      return;
    }

    const processedData = this.dataInfo.dataProvider.process();

    const options =
        this.dataInfo.dataProvider.getOptions() || this.dataInfo.options;

    if (processedData) {
      // tslint:disable-next-line:no-any
      this.chart.draw(processedData, options as any);
    }

    this.processedNumberOfRows.emit(
        processedData ? processedData.getNumberOfRows() : 0);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['corechart', 'table']});
    google.charts.setOnLoadCallback(() => {
      this.initChart();
      this.initDataProvider();
      this.draw();
    });
  }

  initChart() {
    switch (this.chartType) {
      case ChartType.AREA_CHART:
        this.chart =
            new google.visualization.AreaChart(this.elementRef.nativeElement);
        break;
      case ChartType.BAR_CHART:
        this.chart =
            new google.visualization.BarChart(this.elementRef.nativeElement);
        break;
      case ChartType.BUBBLE_CHART:
        this.chart =
            new google.visualization.BubbleChart(this.elementRef.nativeElement);
        break;
      case ChartType.CANDLESTICK_CHART:
        this.chart = new google.visualization.CandlestickChart(
            this.elementRef.nativeElement);
        break;
      case ChartType.COLUMN_CHART:
        this.chart =
            new google.visualization.ColumnChart(this.elementRef.nativeElement);
        break;
      case ChartType.COMBO_CHART:
        this.chart =
            new google.visualization.ComboChart(this.elementRef.nativeElement);
        break;
      case ChartType.HISTOGRAM:
        this.chart =
            new google.visualization.Histogram(this.elementRef.nativeElement);
        break;
      case ChartType.LINE_CHART:
        this.chart =
            new google.visualization.LineChart(this.elementRef.nativeElement);
        break;
      case ChartType.PIE_CHART:
        this.chart =
            new google.visualization.PieChart(this.elementRef.nativeElement);
        break;
      case ChartType.SCATTER_CHART:
        this.chart = new google.visualization.ScatterChart(
            this.elementRef.nativeElement);
        break;
      case ChartType.STEPPED_AREA_CHART:
        this.chart = new google.visualization.SteppedAreaChart(
            this.elementRef.nativeElement);
        break;
      case ChartType.TABLE:
        this.chart =
            new google.visualization.Table(this.elementRef.nativeElement);
        break;
      default:
        this.chart = undefined;
        break;
    }
  }

  initDataProvider() {
    if (!this.dataInfo || !this.dataInfo.dataProvider) {
      return;
    }

    if (this.chart) {
      this.dataInfo.dataProvider.setChart(this.chart);
    }
    this.dataInfo.dataProvider.setData(this.dataInfo);
    this.dataInfo.dataProvider.setSortColumns(this.dataInfo.sortColumns || []);
    this.dataInfo.dataProvider.setFilters(this.dataInfo.filters || []);
    this.dataInfo.dataProvider.setUpdateEventListener(() => {
      this.draw();
    });
  }
}

@NgModule({declarations: [Chart], exports: [Chart]})
export class ChartModule {
}
