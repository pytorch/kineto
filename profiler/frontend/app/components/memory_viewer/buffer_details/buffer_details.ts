import {Component} from '@angular/core';
import {Store} from '@ngrx/store';

import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {getActiveHeapObjectState} from 'org_xprof/frontend/app/store/selectors';

/** A buffer details view component. */
@Component({
  selector: 'buffer-details',
  templateUrl: './buffer_details.ng.html',
  styleUrls: ['./buffer_details.scss']
})
export class BufferDetails {
  heapObject: HeapObject|null = null;
  instructionName?: string;
  opcode?: string;
  size?: string;
  unpaddedSize?: string;
  padding?: string;
  expansion?: string;
  shape?: string;
  tfOpName?: string;
  groupName?: string;
  color?: string;

  constructor(private readonly store: Store<{}>) {
    this.store.select(getActiveHeapObjectState)
        .subscribe((heapObject: HeapObject|null) => {
          this.update(heapObject);
        });
  }

  update(heapObject: HeapObject|null) {
    this.heapObject = heapObject;
    if (!heapObject) {
      return;
    }
    const sizeMiB = heapObject.sizeMiB || 0;
    const unpaddedSizeMiB = heapObject.unpaddedSizeMiB || 0;
    this.instructionName = heapObject.instructionName;
    this.opcode = heapObject.opcode ? heapObject.opcode + ' operation' : '';
    this.size = sizeMiB.toFixed(1);
    this.shape = heapObject.shape;
    this.tfOpName = heapObject.tfOpName;
    this.groupName = heapObject.groupName;
    if (unpaddedSizeMiB) {
      this.unpaddedSize = unpaddedSizeMiB.toFixed(1);
      const utilization = unpaddedSizeMiB / sizeMiB;
      this.color = utils.flameColor(utilization, 0.7);
      if (utilization < 1) {
        this.expansion = (1 / utilization).toFixed(1);
        this.padding = (sizeMiB - unpaddedSizeMiB).toFixed(1);
      } else {
        this.expansion = '';
        this.padding = '';
      }
    } else {
      this.unpaddedSize = '';
      this.padding = '';
      this.expansion = '';
      this.color = 'rgb(192,192,192)';
    }
  }
}
