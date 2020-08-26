import {AfterViewInit, ChangeDetectorRef, Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {MemoryProfileProtoOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A memory profile summary view component. */
@Component({
  selector: 'memory-profile-summary',
  templateUrl: './memory_profile_summary.ng.html',
  styleUrls: ['./memory_profile_summary.scss']
})
export class MemoryProfileSummary implements AfterViewInit, OnChanges {
  /** The memory profile summary data. */
  @Input() data: MemoryProfileProtoOrNull = null;

  /** The selected memory ID to show memory profile for. */
  @Input() memoryId: string = '';

  constructor(private readonly changeDetector: ChangeDetectorRef) {}

  ngAfterViewInit() {
    this.memoryProfileSummary();
    this.changeDetector.detectChanges();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.memoryProfileSummary();
  }

  memoryProfileSummary() {
    if (!this.data || !this.data.memoryIds || !this.data.memoryIds.length ||
        !this.data.memoryProfilePerAllocator) {
      return;
    }

    const summary =
        this.data.memoryProfilePerAllocator[this.memoryId].profileSummary;
    const snapshots = this.data.memoryProfilePerAllocator[this.memoryId]
                          .memoryProfileSnapshots;
    if (!summary || !snapshots) {
      return;
    }

    const peakStats = summary.peakStats;
    if (!peakStats) {
      return;
    }

    let numAllocations = 0;
    let numDeallocations = 0;
    for (let i = 0; i < snapshots.length; i++) {
      const snapshot = snapshots[i];
      if (!snapshot || !snapshot.activityMetadata ||
          !snapshot.activityMetadata.memoryActivity) {
        return;
      }
      if (snapshot.activityMetadata.memoryActivity === 'ALLOCATION') {
        numAllocations++;
      } else if (snapshot.activityMetadata.memoryActivity === 'DEALLOCATION') {
        numDeallocations++;
      }
    }

    this.numAllocations = numAllocations;
    this.numDeallocations = numDeallocations;
    this.memoryCapacityGB = this.bytesToGiBs(summary.memoryCapacity).toFixed(2);
    this.peakHeapUsageLifetimeGB =
        this.bytesToGiBs(summary.peakBytesUsageLifetime).toFixed(2);
    this.timestampAtPeakMs =
        this.picoToMilli(summary.peakStatsTimePs).toFixed(1);
    this.peakMemUsageProfileGB =
        this.bytesToGiBs(peakStats.peakBytesInUse).toFixed(2);
    this.stackAtPeakGB =
        this.bytesToGiBs(peakStats.stackReservedBytes).toFixed(2);
    this.heapAtPeakGB =
        this.bytesToGiBs(peakStats.heapAllocatedBytes).toFixed(2);
    this.freeAtPeakGB = this.bytesToGiBs(peakStats.freeMemoryBytes).toFixed(2);
    this.fragmentationAtPeakPct =
        ((peakStats.fragmentation || 0) * 100).toFixed(2) + '%';
  }

  bytesToGiBs(stat: string|number|undefined) {
    if (!stat) return 0;
    return Number(stat) / Math.pow(2, 30);
  }

  picoToMilli(timePs: string|undefined) {
    if (!timePs) return 0;
    return Number(timePs) / Math.pow(10, 9);
  }

  title = 'Memory Profile Summary';
  numAllocations = 0;
  numDeallocations = 0;
  memoryCapacityGB = '';
  peakHeapUsageLifetimeGB = '';
  peakMemUsageProfileGB = '';
  timestampAtPeakMs = '';
  stackAtPeakGB = '';
  heapAtPeakGB = '';
  freeAtPeakGB = '';
  fragmentationAtPeakPct = '';
}
