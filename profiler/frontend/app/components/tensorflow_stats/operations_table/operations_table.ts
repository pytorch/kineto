import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {OpExecutor} from 'org_xprof/frontend/app/common/constants/enums';
import {TensorflowStatsDataOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

const DIVIDE_COLUMN = (dataColumnIndex: number, divisorColumnIndex: number) => {
  return (dataTable: google.visualization.DataTable|null,
          rowNum: number): number => {
    if (!dataTable) return 0.0;
    return SAFE_DIVIDE(
        dataTable.getValue(rowNum, dataColumnIndex),
        dataTable.getValue(rowNum, divisorColumnIndex));
  };
};

const SAFE_DIVIDE = (dividend: number, divisor: number): number => {
  if (!divisor) return 0.0;
  return 1.0 * dividend / divisor;
};

const WEIGHTED_COLUMN =
    (dataColumnIndex: number, weightedColumnIndex: number) => {
      return (dataTable: google.visualization.DataTable|null,
              rowNum: number): number => {
        if (!dataTable) return 0.0;
        return 1.0 * dataTable.getValue(rowNum, dataColumnIndex) *
            dataTable.getValue(rowNum, weightedColumnIndex);
      };
    };

const MINIMUM_ROWS = 20;

/** An operations table view component. */
@Component({
  selector: 'operations-table',
  templateUrl: './operations_table.ng.html',
  styleUrls: ['./operations_table.scss']
})
export class OperationsTable implements OnChanges, OnInit {
  /** The tensorflow stats data. */
  @Input() tensorflowStatsData: TensorflowStatsDataOrNull = null;

  /** The Op executor. */
  @Input() opExecutor: OpExecutor = OpExecutor.NONE;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  table: google.visualization.Table|null = null;
  title = '';

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawTable();
  }

  drawTable() {
    if (!this.table || !this.tensorflowStatsData || !this.opExecutor) {
      return;
    }

    const dataView = this.opExecutor === OpExecutor.DEVICE ?
        this.makeDataViewForDevice() :
        this.makeDataViewForHost();
    if (!dataView) {
      return;
    }

    const options = {
      allowHtml: true,
      alternatingRowStyle: false,
      showRowNumber: false,
      width: '100%',
      height: dataView.getNumberOfRows() < MINIMUM_ROWS ? '' : '600px',
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
    };

    this.table.draw(dataView, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.load('current', {'packages': ['table']})
    google.charts.setOnLoadCallback(() => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
  }

  makeDataViewForDevice(): google.visualization.DataView|null {
    if (!this.table || !this.tensorflowStatsData ||
        this.opExecutor !== OpExecutor.DEVICE) {
      return null;
    }

    this.title = 'Device-side TensorFlow operations (grouped by TYPE)';

    let dataTable =
        new google.visualization.DataTable(this.tensorflowStatsData);
    let dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{
      'column': 1,
      'value': 'Device',
    }]));

    /**
     * column 0: type
     * column 1: occurrences
     * column 2: self_time
     * column 3: self_time_on_device
     */
    dataView.setColumns([
      2,
      4,
      7,
      9,
    ]);

    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_device
     */
    dataTable =
        new (google.visualization as any)['data']['group'](dataView, [0], [
          {
            'column': 1,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
          {
            'column': 2,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
          {
            'column': 3,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
        ]);
    dataTable.sort({
      'column': 2,
      'desc': true,
    });
    const decimalFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    decimalFormatter.format(dataTable, 2);
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(dataTable, 3);

    dataView = new google.visualization.DataView(dataTable);
    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_device
     */
    dataView.setColumns([
      0,
      1,
      2,
      3,
    ]);

    return dataView;
  }

  makeDataViewForHost(): google.visualization.DataView|null {
    if (!this.table || !this.tensorflowStatsData ||
        this.opExecutor !== OpExecutor.HOST) {
      return null;
    }

    this.title = 'Host-side TensorFlow operations (grouped by TYPE)';

    let dataTable =
        new google.visualization.DataTable(this.tensorflowStatsData);
    let dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{
      'column': 1,
      'value': 'Host',
    }]));

    /**
     * column 0: type
     * column 1: occurrences
     * column 2: self_time
     * column 3: self_time_on_host
     */
    dataView.setColumns([2, 4, 7, 11]);

    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_host
     */
    dataTable =
        new (google.visualization as any)['data']['group'](dataView, [0], [
          {
            'column': 1,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
          {
            'column': 2,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
          {
            'column': 3,
            'aggregation': (google.visualization as any)['data']['sum'],
            'type': 'number',
          },
        ]);
    dataTable.sort({
      'column': 2,
      'desc': true,
    });
    const decimalFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    decimalFormatter.format(dataTable, 2);
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(dataTable, 3);

    return new google.visualization.DataView(dataTable);
  }
}
