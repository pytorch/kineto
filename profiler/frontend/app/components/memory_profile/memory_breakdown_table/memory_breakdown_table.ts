import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';

import {MemoryProfileProtoOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

const DATA_TABLE_OPERATION_INDEX = 0;

/** A memory breakdown table view component. */
@Component({
  selector: 'memory-breakdown-table',
  templateUrl: './memory_breakdown_table.ng.html',
  styleUrls: ['./memory_breakdown_table.scss']
})
export class MemoryBreakdownTable implements OnChanges, OnInit {
  /** The memory profile proto data. */
  @Input() memoryProfileData: MemoryProfileProtoOrNull = null;

  /** The selected memory ID to show memory profile for. */
  @Input() memoryId: string = '';

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  dataTable: google.visualization.DataTable|null = null;
  filterOperation: string = '';
  table: google.visualization.Table|null = null;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataTable = null;
    this.drawTable();
  }

  createDataTable() {
    if (!this.table || !this.memoryProfileData ||
        !this.memoryProfileData.memoryProfilePerAllocator || !!this.dataTable) {
      return;
    }

    this.dataTable = new google.visualization.DataTable();
    this.dataTable.addColumn('string', 'Op Name');
    this.dataTable.addColumn('number', 'Allocation Size (GiBs)');
    this.dataTable.addColumn('number', 'Requested Size (GiBs)');
    this.dataTable.addColumn('number', 'Occurrences');
    this.dataTable.addColumn('string', 'Region type');
    this.dataTable.addColumn('string', 'Data type');
    this.dataTable.addColumn('string', 'Shape');

    const snapshots =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .memoryProfileSnapshots;
    const activeAllocations =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .activeAllocations;
    const specialAllocations =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .specialAllocations;
    if (!snapshots || !activeAllocations || !specialAllocations) {
      return;
    }

    for (let i = 0; i < activeAllocations.length; i++) {
      const index = Number(activeAllocations[i].snapshotIndex);
      const specialIndex = Number(activeAllocations[i].specialIndex);
      // Use snapshot index or special index, whichever is positve.
      let metadata;
      if (index >= 0) {
        // It may be dropped depending on the max_num_snapshots query parameter
        // which is set to 1000 by default.
        if (!(index in snapshots)) continue;
        metadata = snapshots[index].activityMetadata;
      } else {
        metadata = specialAllocations[specialIndex];
      }
      if (!metadata) {
        continue;
      }
      this.dataTable.addRow([
        metadata.tfOpName,
        this.bytesToGiBs(metadata.allocationBytes),
        this.bytesToGiBs(metadata.requestedBytes),
        Number(activeAllocations[i].numOccurrences),
        metadata.regionType,
        metadata.dataType,
        metadata.tensorShape,
      ]);
    }

    const decimalPtFormatter =
        new google.visualization.NumberFormat({fractionDigits: 3});
    decimalPtFormatter.format(this.dataTable, 1); /* requested_size */
    decimalPtFormatter.format(this.dataTable, 2); /* allocation_size */
  }

  drawTable() {
    if (!this.table || !this.memoryProfileData) {
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
    };

    this.table.draw(dataView, options as google.visualization.TableOptions);
  }

  getDataView(): google.visualization.DataView|null {
    if (!this.dataTable) {
      this.createDataTable();
    }

    const dataTable = this.getFilteredDataTable();
    if (!dataTable) {
      return null;
    }

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{column: 2, minValue: 0.001}]));
    return dataView;
  }

  getFilteredDataTable(): google.visualization.DataTable|null {
    if (!this.dataTable) {
      return null;
    }

    /* tslint:disable no-any */
    const filters = [];
    if (this.filterOperation.trim()) {
      const filter = this.filterOperation.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_OPERATION_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      } as any);
    }
    /* tslint:enable */

    if (filters.length > 0) {
      const dataView = new google.visualization.DataView(this.dataTable);
      dataView.setRows(this.dataTable.getFilteredRows(filters));
      return dataView.toDataTable();
    }

    return this.dataTable;
  }

  bytesToGiBs(stat: string|number|undefined) {
    if (!stat) return 0;
    return Number(stat) / Math.pow(2, 30);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    // tslint:disable-next-line:no-any
    (google.charts as any)['load']('current', {'packages': ['table']});
    google.charts.setOnLoadCallback(() => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
  }
}
