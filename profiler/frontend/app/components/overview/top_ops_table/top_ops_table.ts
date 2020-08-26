import {AfterViewInit, Component, ElementRef, Input, OnChanges, SimpleChanges, ViewChild} from '@angular/core';

import {GeneralAnalysis, InputPipelineAnalysis, TopOpsColumn} from 'org_xprof/frontend/app/common/interfaces/data_table';

const FLOP_RATE_COLUMN_INDEX = 4;

/** A top ops table view component. */
@Component({
  selector: 'top-ops-table',
  templateUrl: './top_ops_table.ng.html',
  styleUrls: ['./top_ops_table.scss']
})
export class TopOpsTable implements AfterViewInit, OnChanges {
  /** The general anaysis data. */
  @Input() generalAnalysis: GeneralAnalysis|null = null;

  /** The input pipeline analyis data. */
  @Input() inputPipelineAnalysis: InputPipelineAnalysis|null = null;

  /** Whether the expansion panel is initially open. */
  @Input() initiallyExpanded: boolean = true;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  title = 'Top TensorFlow operations on TPU';
  table: google.visualization.Table|null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawTable();
  }

  drawTable() {
    if (!this.table || !this.generalAnalysis || !this.inputPipelineAnalysis) {
      return;
    }

    const dataTable = new google.visualization.DataTable(this.generalAnalysis);
    if (dataTable.getNumberOfColumns() < 1) {
      return;
    }

    this.inputPipelineAnalysis.p = this.inputPipelineAnalysis.p || {};
    this.title = 'Top ' + String(dataTable.getNumberOfRows()) +
        ' TensorFlow operations on ' +
        (this.inputPipelineAnalysis.p.hardware_type || 'TPU');
    const columns: TopOpsColumn = {
      selfTimePercent: 0,
      cumulativeTimePercent: 0,
      category: 0,
      operation: 0,
      flopRate: 0,
      tcEligibility: 0,
      tcUtilization: 0,
    };
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      switch (dataTable.getColumnId(i)) {
        case 'selfTimePercent':
          columns.selfTimePercent = i;
          break;
        case 'cumulativeTimePercent':
          columns.cumulativeTimePercent = i;
          break;
        case 'category':
          columns.category = i;
          break;
        case 'operation':
          columns.operation = i;
          break;
        case 'flopRate':
          columns.flopRate = i;
          break;
        case 'tcEligibility':
          columns.tcEligibility = i;
          break;
        case 'tcUtilization':
          columns.tcUtilization = i;
          break;
        default:
          break;
      }
    }

    const percentFormatter =
        new google.visualization.NumberFormat({'pattern': '##.#%'});
    percentFormatter.format(dataTable, columns.selfTimePercent);
    percentFormatter.format(dataTable, columns.cumulativeTimePercent);

    const oneDecimalPtFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 1});
    oneDecimalPtFormatter.format(dataTable, columns.flopRate);

    dataTable.setProperty(0, columns.selfTimePercent, 'style', 'width: 10%');
    dataTable.setProperty(
        0, columns.cumulativeTimePercent, 'style', 'width: 10%');
    dataTable.setProperty(0, columns.category, 'style', 'width: 20%');
    dataTable.setProperty(0, columns.operation, 'style', 'width: 40%');
    dataTable.setProperty(0, columns.flopRate, 'style', 'display: 10%');
    dataTable.setProperty(0, columns.tcEligibility, 'style', 'display: 5%');
    dataTable.setProperty(0, columns.tcUtilization, 'style', 'display: 5%');
    const options = {
      alternatingRowStyle: false,
      showRowNumber: false,
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
      width: '100%',
    };

    if ((this.inputPipelineAnalysis.p.hardware_type || 'TPU') !== 'TPU') {
      dataTable.removeColumn(FLOP_RATE_COLUMN_INDEX);
    }

    this.table.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['table']});
    google.charts.setOnLoadCallback(() => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
  }
}
