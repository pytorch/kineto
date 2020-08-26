import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {SimpleDataTableOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

declare interface KernelStatsColumn {
  rank: number;
  kernelName: number;
  registersPerThread: number;
  shmemBytes: number;
  blockDim: number;
  gridDim: number;
  isKernelUsingTensorCore: number;
  isOpTensorCoreEligible: number;
  opName: number;
  occurrences: number;
  totalDurationUs: number;
  avgDurationUs: number;
  minDurationUs: number;
  maxDurationUs: number;
}

declare interface SortEvent {
  column: number;
  ascending: boolean;
}

const DATA_TABLE_RANK_INDEX = 0;
const DATA_TABLE_KERNEL_NAME_INDEX = 1;
const DATA_TABLE_OP_NAME_INDEX = 8;

/** A kernel stats table view component. */
@Component({
  selector: 'kernel-stats-table',
  templateUrl: './kernel_stats_table.ng.html',
  styleUrls: ['./kernel_stats_table.scss']
})
export class KernelStatsTable implements OnChanges, OnInit {
  /** The kernel stats data. */
  @Input() kernelStatsData: SimpleDataTableOrNull = null;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  columns: KernelStatsColumn = {
    rank: 0,
    kernelName: 0,
    registersPerThread: 0,
    shmemBytes: 0,
    blockDim: 0,
    gridDim: 0,
    isKernelUsingTensorCore: 0,
    isOpTensorCoreEligible: 0,
    opName: 0,
    occurrences: 0,
    totalDurationUs: 0,
    avgDurationUs: 0,
    minDurationUs: 0,
    maxDurationUs: 0,
  };
  dataTable: google.visualization.DataTable|null = null;
  filterKernelName = '';
  filterOpName = '';
  sortAscending = true;
  sortColumn = 0;
  table: google.visualization.Table|null = null;

  loading = true;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataTable = null;
    this.drawTable();
  }

  createDataTable() {
    if (!this.table || !this.kernelStatsData || !!this.dataTable) {
      return;
    }

    this.dataTable = new google.visualization.DataTable(this.kernelStatsData);
  }

  drawTable() {
    if (!this.table || !this.kernelStatsData) {
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
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
      sort: 'event',
      sortAscending: this.sortAscending,
      sortColumn: this.sortColumn,
      page: 'enable',
      pageSize: 100,
    };

    this.table.draw(dataView, options as google.visualization.TableOptions);
  }

  enumerateColumns() {
    if (!this.dataTable) {
      return;
    }

    for (let i = 0; i < this.dataTable.getNumberOfColumns(); i++) {
      switch (this.dataTable.getColumnId(i)) {
        case 'rank':
          this.columns.rank = i;
          break;
        case 'kernel_name':
          this.columns.kernelName = i;
          break;
        case 'registers_per_thread':
          this.columns.registersPerThread = i;
          break;
        case 'shmem_bytes':
          this.columns.shmemBytes = i;
          break;
        case 'block_dim':
          this.columns.blockDim = i;
          break;
        case 'grid_dim':
          this.columns.gridDim = i;
          break;
        case 'is_kernel_using_tensor_core':
          this.columns.isKernelUsingTensorCore = i;
          break;
        case 'is_op_tensor_core_eligible':
          this.columns.isOpTensorCoreEligible = i;
          break;
        case 'op_name':
          this.columns.opName = i;
          break;
        case 'occurrences':
          this.columns.occurrences = i;
          break;
        case 'total_duration_us':
          this.columns.totalDurationUs = i;
          break;
        case 'avg_duration_us':
          this.columns.avgDurationUs = i;
          break;
        case 'min_duration_us':
          this.columns.minDurationUs = i;
          break;
        case 'max_duration_us':
          this.columns.maxDurationUs = i;
          break;
        default:
          break;
      }
    }
  }

  getDataView(): google.visualization.DataView|null {
    if (!this.dataTable) {
      this.createDataTable();
      this.enumerateColumns();
    }

    if (this.dataTable) {
      this.dataTable.setProperty(
          0, this.columns.kernelName, 'style', 'width: 30%');
      this.dataTable.setProperty(0, this.columns.opName, 'style', 'width: 25%');
    }

    const dataView = this.sortDataView();
    if (!dataView) {
      return null;
    }
    dataView.setColumns([
      this.columns.rank,
      this.columns.kernelName,
      this.columns.registersPerThread,
      this.columns.shmemBytes,
      this.columns.blockDim,
      this.columns.gridDim,
      this.columns.isKernelUsingTensorCore,
      this.columns.isOpTensorCoreEligible,
      this.columns.opName,
      this.columns.occurrences,
      this.columns.totalDurationUs,
      this.columns.avgDurationUs,
      this.columns.minDurationUs,
      this.columns.maxDurationUs,
    ]);

    return dataView;
  }

  getFilteredDataTable(): google.visualization.DataTable|null {
    if (!this.dataTable) {
      return null;
    }

    const filters = [];
    if (this.filterKernelName.trim()) {
      const filter = this.filterKernelName.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_KERNEL_NAME_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
        // tslint:disable-next-line:no-any
      } as any);
    }
    if (this.filterOpName.trim()) {
      const filter = this.filterOpName.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_OP_NAME_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
        // tslint:disable-next-line:no-any
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

    google.charts.load('current', {'packages': ['table']});
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

    const sortedIndex = dataTable.getSortedRows({
      column: this.sortColumn,
      desc: !this.sortAscending,
    });
    if (this.sortColumn === DATA_TABLE_RANK_INDEX && !this.sortAscending) {
      let n = sortedIndex.length;
      sortedIndex.forEach((v) => {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, n--);
      });
    } else {
      sortedIndex.forEach((v, idx) => {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, idx + 1);
      });
    }

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(sortedIndex);

    return dataView;
  }
}
