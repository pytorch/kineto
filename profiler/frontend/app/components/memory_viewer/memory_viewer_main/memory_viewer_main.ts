import {Component, Input, OnChanges, OnDestroy, SimpleChanges} from '@angular/core';
import {Store} from '@ngrx/store';
import {BufferAllocationInfo} from 'org_xprof/frontend/app/common/interfaces/buffer_allocation_info';
import {HloProtoOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {MemoryUsage} from 'org_xprof/frontend/app/components/memory_viewer/memory_usage/memory_usage';
import {setActiveHeapObjectAction, setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

interface BufferSpan {
  alloc: number;
  free: number;
}

/** A memory viewer component. */
@Component({
  selector: 'memory-viewer-main',
  templateUrl: './memory_viewer_main.ng.html',
  styleUrls: ['./memory_viewer_main.scss']
})
export class MemoryViewerMain implements OnDestroy, OnChanges {
  /** XLA Hlo proto */
  @Input() hloProto: HloProtoOrNull = null;

  /** XLA memory space color */
  @Input() memorySpaceColor: number = 0;

  moduleName: string = '';
  peakInfo?: BufferAllocationInfo;
  activeInfo?: BufferAllocationInfo;
  peakHeapSizeMiB: string = '';
  unpaddedPeakHeapSizeMiB: string = '';
  usage?: MemoryUsage;
  heapSizes?: number[];
  maxHeap?: HeapObject[];
  maxHeapBySize?: HeapObject[];
  selectedIndex: number = -1;
  selectedIndexBySize: number = -1;
  unpaddedHeapSizes?: number[];
  hasHeapSimulatorTrace = false;
  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};

  constructor(private readonly store: Store<{}>) {}

  ngOnChanges(changes: SimpleChanges) {
    this.update();
  }

  ngOnDestroy() {
    this.dispatchActiveHeapObject();
  }

  private dispatchActiveHeapObject(heapObject: HeapObject|null = null) {
    this.store.dispatch(
        setActiveHeapObjectAction({activeHeapObject: heapObject}));
    if (heapObject) {
      const span = this.getLogicalBufferSpan(heapObject.logicalBufferId);
      this.activeInfo = {
        size: heapObject.sizeMiB || 0,
        alloc: span.alloc,
        free: span.free,
        color: utils.getChartItemColorByIndex(heapObject.color || 0),
      };
    } else {
      this.activeInfo = undefined;
      this.selectedIndex = -1;
      this.selectedIndexBySize = -1;
    }
  }

  private getLogicalBufferSpan(index?: number): BufferSpan {
    const bufferSpan: BufferSpan = {alloc: 0, free: 0};
    if (index && this.usage && this.usage.logicalBufferSpans &&
        this.heapSizes) {
      const span = this.usage.logicalBufferSpans[index];
      if (span) {
        bufferSpan.alloc = span[0];
        bufferSpan.free = span[1] < 0 ? this.heapSizes.length - 1 : span[1];
      } else {
        bufferSpan.free = this.heapSizes.length - 1;
      }
    }
    return bufferSpan;
  }

  setSelectedHepObject(selectedIndex: number) {
    if (!this.usage) {
      return;
    }
    if (selectedIndex === -1) {
      this.dispatchActiveHeapObject();
    } else {
      this.dispatchActiveHeapObject(this.usage.maxHeap[selectedIndex]);
      this.selectedIndexBySize = this.usage.maxHeapToBySize[selectedIndex];
    }
  }

  setSelectedHepObjectBySize(selectedIndexBySize: number) {
    if (!this.usage) {
      return;
    }
    if (selectedIndexBySize === -1) {
      this.dispatchActiveHeapObject();
    } else {
      this.dispatchActiveHeapObject(
          this.usage.maxHeapBySize[selectedIndexBySize]);
      this.selectedIndex = this.usage.bySizeToMaxHeap[selectedIndexBySize];
    }
  }

  update() {
    const data = this.hloProto;
    this.diagnostics.errors = [];
    if (!data || !data.hloModule) {
      this.diagnostics.errors.push(
          'We fail to fetch a valid input. The input is empty or too large.');
      return;
    }
    if (!data.bufferAssignment) {
      this.diagnostics.errors.push(
          'The HloProto does not contain a buffer assignment. ' +
          'Therefore, we don\'t know the memory usage.');
      return;
    }
    this.moduleName = data.hloModule.name || '';
    this.usage = new MemoryUsage(data, this.memorySpaceColor);
    this.peakHeapSizeMiB =
        utils.bytesToMiB(this.usage.peakHeapSizeBytes).toFixed(2);
    this.unpaddedPeakHeapSizeMiB =
        utils.bytesToMiB(this.usage.unpaddedPeakHeapSizeBytes).toFixed(2);
    this.heapSizes = this.usage.heapSizes;
    this.unpaddedHeapSizes = this.usage.unpaddedHeapSizes;
    this.peakInfo = {
      size: utils.bytesToMiB(this.usage.peakHeapSizeBytes),
      alloc: this.usage.peakHeapSizePosition + 1,
      free: this.usage.peakHeapSizePosition + 2,
    };
    this.maxHeap = this.usage.maxHeap;
    this.maxHeapBySize = this.usage.maxHeapBySize;

    this.hasHeapSimulatorTrace = !!this.heapSizes && this.heapSizes.length > 0;
  }
}
