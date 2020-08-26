import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';

import {BufferAllocationInfo} from 'org_xprof/frontend/app/common/interfaces/buffer_allocation_info';

/** A program order chart view component. */
@Component({
  selector: 'program-order-chart',
  templateUrl: './program_order_chart.ng.html',
  styleUrls: ['./program_order_chart.scss']
})
export class ProgramOrderChart implements OnChanges, OnInit {
  /** The heap size list. */
  @Input() heapSizes?: number[];

  /** The unpadded heap size list. */
  @Input() unpaddedHeapSizes?: number[];

  /** The peak buffer allocation information. */
  @Input() peakInfo?: BufferAllocationInfo;

  /** The active buffer allocation information. */
  @Input() activeInfo?: BufferAllocationInfo;

  @ViewChild('activeChart', {static: false}) activeChartRef!: ElementRef;
  @ViewChild('chart', {static: false}) chartRef!: ElementRef;
  @ViewChild('peakChart', {static: false}) peakChartRef!: ElementRef;

  activeChart: google.visualization.AreaChart|null = null;
  chart: google.visualization.LineChart|null = null;
  peakChart: google.visualization.AreaChart|null = null;
  maxSize: number = 0;
  maxOrder: number = 0;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['heapSizes'] || changes['unpaddedHeapSizes']) {
      this.drawChart();
    }
    if (changes['peakInfo']) {
      this.drawPeakChart();
    }
    if (changes['activeInfo']) {
      this.drawActiveChart();
    }
  }

  drawActiveChart() {
    if (!this.activeChart) {
      return;
    }

    if (!this.activeInfo) {
      this.activeChart.clearChart();
      return;
    }

    const dataTable = google.visualization.arrayToDataTable([
      ['X', 'Size'],
      [this.activeInfo.alloc, this.activeInfo.size],
      [this.activeInfo.free, this.activeInfo.size],
    ]);

    const options = {
      areaOpacity: 0.7,
      backgroundColor: 'transparent',
      chartArea: {
        left: 50,
        right: 50,
        width: '90%',
        height: '90%',
      },
      colors: [this.activeInfo.color || ''],
      focusTarget: 'none',
      hAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxOrder,
        },
      },
      vAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'none'},
      lineWidth: 2,
    };

    this.activeChart.draw(
        dataTable, options as google.visualization.AreaChartOptions);
  }

  drawChart() {
    if (!this.chart || !this.heapSizes || !this.unpaddedHeapSizes) {
      return;
    }

    const data = [];
    this.maxOrder = this.heapSizes.length - 1;
    this.maxSize = 0;
    for (let i = 0; i < this.heapSizes.length; i++) {
      this.maxSize = Math.max(
          this.maxSize, Math.max(this.heapSizes[i], this.unpaddedHeapSizes[i]));
      data.push([i, this.heapSizes[i], this.unpaddedHeapSizes[i]]);
    }
    this.maxSize = Math.round(this.maxSize * 1.1);

    const dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'X');
    dataTable.addColumn('number', 'Size');
    dataTable.addColumn('number', 'Unpadded Size');
    dataTable.addRows(data);

    const options = {
      backgroundColor: 'transparent',
      chartArea: {
        left: 50,
        right: 50,
        width: '90%',
        height: '90%',
      },
      focusTarget: 'none',
      hAxis: {
        baselineColor: 'transparent',
        viewWindow: {
          min: 0,
          max: this.maxOrder,
        },
      },
      vAxis: {
        baselineColor: 'transparent',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'top'},
    };

    this.chart.draw(
        dataTable, options as google.visualization.LineChartOptions);
  }

  drawPeakChart() {
    if (!this.peakChart || !this.peakInfo) {
      return;
    }

    const peakWidth = Math.max(Math.round(this.maxOrder / 50), 1);
    const peakAlloc =
        Math.max(Math.round(this.peakInfo.alloc - peakWidth / 2), 0);
    const peakFree = Math.min(peakAlloc + peakWidth, this.maxOrder);
    const dataTable = google.visualization.arrayToDataTable([
      ['X', 'Size'],
      [peakAlloc, this.peakInfo.size],
      [peakFree, this.peakInfo.size],
    ]);

    const options = {
      backgroundColor: 'transparent',
      chartArea: {
        left: 50,
        right: 50,
        width: '90%',
        height: '90%',
      },
      colors: ['#00ff00'],
      focusTarget: 'none',
      hAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxOrder,
        },
      },
      vAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'none'},
      lineWidth: 0,
    };

    this.peakChart.draw(
        dataTable, options as google.visualization.AreaChartOptions);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      this.activeChart =
          new google.visualization.AreaChart(this.activeChartRef.nativeElement);
      this.chart =
          new google.visualization.LineChart(this.chartRef.nativeElement);
      this.peakChart =
          new google.visualization.AreaChart(this.peakChartRef.nativeElement);
      this.drawChart();
      this.drawPeakChart();
      this.drawActiveChart();
    });
  }
}
