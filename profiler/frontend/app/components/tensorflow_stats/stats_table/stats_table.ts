import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {TensorflowStatsDataOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

declare interface SortEvent {
  column: number;
  ascending: boolean;
}

const DATA_TABLE_RANK_INDEX = 1;
const DATA_TABLE_EXECUTOR_INDEX = 2;
const DATA_TABLE_TYPE_INDEX = 3;
const DATA_TABLE_OPERATION_INDEX = 4;
const DATA_TABLE_DEVICE_PERCENT_INDEX = 10;
const DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX = 11;
const DATA_TABLE_HOST_PERCENT_INDEX = 12;
const DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX = 13;
const DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX = 10;
const DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX = 12;
const MAXIMUM_ROWS = 1000;

/** A stats table view component. */
@Component({
  selector: 'stats-table',
  templateUrl: './stats_table.ng.html',
  styleUrls: ['./stats_table.scss']
})
export class StatsTable implements OnChanges, OnInit {
  /** The tensorflow stats data. */
  @Input() tensorflowStatsData: TensorflowStatsDataOrNull = null;

  /** The property indicating whether the device data exists. */
  @Input() hasDeviceData = false;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  dataTable: google.visualization.DataTable|null = null;
  filterExecutor = '';
  filterType = '';
  filterOperation = '';
  sortAscending = true;
  sortColumn = -1;
  table: google.visualization.Table|null = null;
  totalOperations = '';

  loading = true;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataTable = null;
    this.drawTable();
  }

  createDataTable() {
    if (!this.table || !this.tensorflowStatsData || !!this.dataTable) {
      return;
    }

    const dataTable =
        new google.visualization.DataTable(this.tensorflowStatsData);
    const totalOperations = dataTable.getNumberOfRows();
    this.dataTable = dataTable.clone();
    this.totalOperations = '';
    if (totalOperations > MAXIMUM_ROWS) {
      this.totalOperations = String(totalOperations);
      this.dataTable.removeRows(MAXIMUM_ROWS, totalOperations - MAXIMUM_ROWS);
    }

    const zeroDecimalPtFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    zeroDecimalPtFormatter.format(this.dataTable, 5); /** total_time */
    zeroDecimalPtFormatter.format(this.dataTable, 6); /** avg_time */
    zeroDecimalPtFormatter.format(this.dataTable, 7); /** total_self_time */
    zeroDecimalPtFormatter.format(this.dataTable, 8); /** avg_self_time */

    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(
        this.dataTable, 9); /** device_total_self_time_percent */
    percentFormatter.format(
        this.dataTable, 11); /** host_total_self_time_percent */

    /**
     * Format tensorcore utilization column if it exists in dataTable.
     * This column does not exist in dataTable if the device is not GPU.
     */
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      if (this.dataTable.getColumnId(i) === 'gpu_tensorcore_utilization') {
        percentFormatter.format(this.dataTable, i);
        break;
      }
    }

    this.dataTable.insertColumn(1, 'number', 'Rank');
  }

  drawTable() {
    if (!this.table || !this.tensorflowStatsData) {
      return;
    }

    const dataView = this.getDataView();
    if (!dataView) {
      return;
    }

    const options = {
      allowHtml: true,
      alternatingRowStyle: false,
      showRowNumber: false,
      width: '100%',
      height: '600px',
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
      sort: 'event',
      sortAscending: this.sortAscending,
      sortColumn: this.sortColumn,
    };

    this.table.draw(dataView, options as google.visualization.TableOptions);
  }

  getDataView(): google.visualization.DataView|null {
    if (!this.dataTable) {
      this.createDataTable();
    }

    const dataView = this.sortDataView();
    if (!dataView) {
      return null;
    }

    dataView.hideColumns([0]);

    return dataView;
  }

  getFilteredDataTable(): google.visualization.DataTable|null {
    if (!this.dataTable) {
      return null;
    }

    const filters = [];
    if (this.filterExecutor.trim()) {
      const filter = this.filterExecutor.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_EXECUTOR_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      } as any);
    }
    if (this.filterType.trim()) {
      const filter = this.filterType.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_TYPE_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      } as any);
    }
    if (this.filterOperation.trim()) {
      const filter = this.filterOperation.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_OPERATION_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      } as any);
    }

    if (filters.length > 0) {
      const dataView = new google.visualization.DataView(this.dataTable);
      dataView.setRows(this.dataTable.getFilteredRows(filters));
      return dataView.toDataTable();
    }

    return this.dataTable;
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
      google.visualization.events.addListener(
          this.table, 'sort', (event: SortEvent) => {
            this.sortColumn = event.column;
            this.sortAscending = event.ascending;
            this.drawTable();
          });
      this.drawTable();
    });
  }

  sortDataView(): google.visualization.DataView|null {
    const dataTable = this.getFilteredDataTable();
    if (!dataTable) {
      return null;
    }

    let sortColumn = this.sortColumn + 1;
    if (this.sortColumn === 0 ||
        this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ||
        this.sortColumn === DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX) {
      sortColumn = this.sortColumn;
    }

    const sortedIndex = dataTable.getSortedRows({
      column: sortColumn,
      desc: !this.sortAscending,
    });
    let sumOfDevice = 0;
    let sumOfHost = 0;
    if (sortColumn === 0 && !this.sortAscending) {
      let n = sortedIndex.length;
      sortedIndex.forEach((v) => {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, n--);
        sumOfDevice += dataTable.getValue(v, DATA_TABLE_DEVICE_PERCENT_INDEX);
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX, sumOfDevice);
        sumOfHost += dataTable.getValue(v, DATA_TABLE_HOST_PERCENT_INDEX);
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX, sumOfHost);
      });
    } else {
      sortedIndex.forEach((v, idx) => {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, idx + 1);
        sumOfDevice += dataTable.getValue(v, DATA_TABLE_DEVICE_PERCENT_INDEX);
        if (sumOfDevice > 100) {
          sumOfDevice = 100;
        }
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX, sumOfDevice);
        sumOfHost += dataTable.getValue(v, DATA_TABLE_HOST_PERCENT_INDEX);
        if (sumOfHost > 100) {
          sumOfHost = 100;
        }
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX, sumOfHost);
      });
      if (!this.sortAscending &&
          (this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ||
           this.sortColumn === DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX)) {
        let sumOfPercent =
            this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ?
            sumOfDevice :
            sumOfHost;
        const index =
            this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ?
            DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX :
            DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX;
        sortedIndex.forEach((v) => {
          dataTable.setCell(v, index, sumOfPercent);
          sumOfPercent -= dataTable.getValue(v, index - 1);
          if (sumOfPercent < 0) {
            sumOfPercent = 0;
          }
        });
      }
    }
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(
        dataTable, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX);
    percentFormatter.format(
        dataTable, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX);

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(sortedIndex);

    return dataView;
  }
}
