import * as proto from 'org_xprof/frontend/app/common/interfaces/hlo.proto';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {BufferAllocation} from 'org_xprof/frontend/app/components/memory_viewer/xla/buffer_allocation';
import {HloInstruction} from 'org_xprof/frontend/app/components/memory_viewer/xla/hlo_instruction';
import {LogicalBuffer} from 'org_xprof/frontend/app/components/memory_viewer/xla/logical_buffer';
import {Shape} from 'org_xprof/frontend/app/components/memory_viewer/xla/shape';

interface MemoryUsageBytes {
  padded: number;
  unpadded: number;
}

/**
 * Provides calculation of memory usage from xla buffer assignment.
 * @final
 */
export class MemoryUsage {
  private readonly buffers: LogicalBuffer[];
  private readonly idToBuffer: {[key: number]: LogicalBuffer};
  private readonly idToBufferAllocation: {[key: number]: BufferAllocation};
  private readonly nameToHlo: {[key: string]: HloInstruction};
  private readonly unSeenLogicalBuffers: Set<number>;
  private readonly seenBufferAllocations: Set<number>;
  private nColor: number;
  private rest: number;
  private memorySpaceColor: number;

  peakHeapSizeBytes: number;
  unpaddedPeakHeapSizeBytes: number;
  peakLogicalBuffers: number[];
  peakHeapSizePosition: number;
  indefiniteMemoryUsageBytes: MemoryUsageBytes;
  heapSizes: number[];
  unpaddedHeapSizes: number[];
  maxHeap: HeapObject[];
  maxHeapBySize: HeapObject[];
  bySizeToMaxHeap: number[];
  maxHeapToBySize: number[];
  logicalBufferSpans: {[key: number]: number[]};

  smallBufferSize: number;

  constructor(hloProto: proto.HloProto, memorySpaceColor: number) {
    this.buffers = [];
    this.idToBuffer = {};
    this.idToBufferAllocation = {};
    this.nameToHlo = {};
    this.unSeenLogicalBuffers = new Set();
    this.seenBufferAllocations = new Set();
    this.nColor = 0;
    this.rest = 0;
    this.memorySpaceColor = memorySpaceColor;

    this.peakHeapSizeBytes = 0;
    this.unpaddedPeakHeapSizeBytes = 0;
    this.peakLogicalBuffers = [];
    this.peakHeapSizePosition = 0;
    this.indefiniteMemoryUsageBytes = {padded: 0, unpadded: 0};
    this.heapSizes = [];
    this.unpaddedHeapSizes = [];
    this.maxHeap = [];
    this.maxHeapBySize = [];
    this.bySizeToMaxHeap = [];
    this.maxHeapToBySize = [];
    this.logicalBufferSpans = {};
    this.smallBufferSize = 16 * 1024;

    this.initHloInstructions(hloProto.hloModule);
    this.initMemoryUsage(memorySpaceColor, hloProto.bufferAssignment);
    this.initMaxHeap();
  }

  /**
   * Adds the logical buffer as an element in the maxHeap with constitutent
   * logical buffers.
   * @private
   */
  private addHeapObject(buffer: LogicalBuffer, groupName: string) {
    if (!buffer) {
      return;
    }
    if (buffer.size <= this.smallBufferSize) {
      this.rest += buffer.size;
      return;
    }
    if (!buffer.instructionName) {
      return;
    }
    const inst = this.nameToHlo[buffer.instructionName];
    if (!inst) {
      return;
    }
    if (!inst.shape) {
      return;
    }
    const shape = inst.shape.resolveShapeIndex(buffer.shapeIndex);
    if (!shape) {
      return;
    }
    const heapObject = this.newHeapObject(buffer, shape, inst, groupName);
    if (heapObject) {
      this.maxHeap.push(heapObject);
    }
  }

  /**
   * Calculate the indefinite memory usage from the unseen logical buffers.
   * Assume they have indefinite lifetime if they are not in thread-local buffer
   * allocations.
   * @private
   */
  private findIndefiniteMemoryUsage(buffers: Set<number>, color: number):
      MemoryUsageBytes {
    const usageBytes: MemoryUsageBytes = {padded: 0, unpadded: 0};
    buffers.forEach(id => {
      const alloc = this.idToBufferAllocation[id];
      if (!alloc || alloc.isThreadLocal) {
        return;
      }
      if (!this.seenBufferAllocations.has(alloc.index)) {
        const buffer = this.idToBuffer[id];
        if (!buffer || buffer.color !== color) return;
        this.seenBufferAllocations.add(alloc.index);
        usageBytes.padded += alloc.size;
        let shape = null;
        if (buffer.instructionName) {
          const hlo = this.nameToHlo[buffer.instructionName];
          if (hlo && hlo.shape) {
            shape = hlo.shape.resolveShapeIndex(buffer.shapeIndex);
          }
        }
        usageBytes.unpadded +=
            shape ? shape.unpaddedHeapSizeBytes() : alloc.size;
        this.addHeapObject(this.idToBuffer[id], alloc.groupName);
      }
    });
    this.indefiniteMemoryUsageBytes = usageBytes;
    return usageBytes;
  }

  /**
   * Finds the peak memory usage from the `HeapSimulatorTrace`.
   * @private
   */
  private findPeakMemoryUsage(trace: proto.HeapSimulatorTrace, color: number) {
    if (!trace) {
      return;
    }
    const heapSizes: number[] = [];
    const unpaddedHeapSizes: number[] = [];
    let logicalBuffers: number[] = [];
    let peakLogicalBuffers: number[] = [];
    let heapSizeBytes = 0;
    let unpaddedHeapSizeBytes = 0;
    let peakHeapSizeBytes = 0;
    let unpaddedPeakHeapSizeBytes = 0;
    let peakHeapSizePosition = 0;

    for (const event of trace.events || []) {
      heapSizes.push(utils.bytesToMiB(heapSizeBytes));
      unpaddedHeapSizes.push(utils.bytesToMiB(unpaddedHeapSizeBytes));
      const eventId = utils.toNumber(event.bufferId);
      const buffer = this.idToBuffer[eventId];
      this.unSeenLogicalBuffers.delete(eventId);
      const alloc = this.idToBufferAllocation[eventId];
      if (alloc) {
        this.seenBufferAllocations.add(alloc.index);
      }
      let shape: Shape|null = null;
      if (buffer.instructionName && buffer.instructionName !== '') {
        const hlo = this.nameToHlo[buffer.instructionName];
        if (hlo && hlo.shape) {
          shape = hlo.shape.resolveShapeIndex(buffer.shapeIndex);
        }
      }
      switch (event.kind) {
        case 'ALLOC':
        case 'SHARE_WITH':
          logicalBuffers.push(eventId);
          heapSizeBytes += buffer.size;
          if (shape) {
            unpaddedHeapSizeBytes += shape.unpaddedHeapSizeBytes();
          }
          this.logicalBufferSpans[eventId] = [heapSizes.length - 1, -1];
          if (heapSizeBytes > peakHeapSizeBytes) {
            peakHeapSizeBytes = heapSizeBytes;
            unpaddedPeakHeapSizeBytes = unpaddedHeapSizeBytes;
            peakHeapSizePosition = heapSizes.length - 1;
            peakLogicalBuffers = logicalBuffers.slice();
          }
          break;
        case 'FREE':
          logicalBuffers = logicalBuffers.filter(item => {
            return item !== eventId;
          });
          heapSizeBytes -= buffer.size;
          if (shape) {
            unpaddedHeapSizeBytes -= shape.unpaddedHeapSizeBytes();
          }
          if (!this.logicalBufferSpans[eventId]) {
            // The logical buffer is not allocated in this module.
            this.logicalBufferSpans[eventId] = [0, heapSizes.length - 1];
            console.warn(event, ' is freed but has seen no allocation event.');
          } else {
            this.logicalBufferSpans[eventId][1] = heapSizes.length - 1;
          }
          if (heapSizeBytes < 0) {
            console.error('heap_size_bytes < 0');
          }
          break;
        default:
          console.log('ERROR: unknown heap event kind:' + event.toString());
          break;
      }
    }

    heapSizes.push(utils.bytesToMiB(heapSizeBytes));
    unpaddedHeapSizes.push(utils.bytesToMiB(unpaddedHeapSizeBytes));
    const indefiniteMemoryUsageBytes =
        this.findIndefiniteMemoryUsage(this.unSeenLogicalBuffers, color);
    this.peakHeapSizeBytes =
        peakHeapSizeBytes + indefiniteMemoryUsageBytes.padded;
    this.unpaddedPeakHeapSizeBytes =
        unpaddedPeakHeapSizeBytes + indefiniteMemoryUsageBytes.unpadded;
    this.peakLogicalBuffers = peakLogicalBuffers;
    this.peakHeapSizePosition = peakHeapSizePosition;
    const addendPadded = utils.bytesToMiB(indefiniteMemoryUsageBytes.padded);
    this.heapSizes = heapSizes.map(item => item + addendPadded);
    const addendUnpadded =
        utils.bytesToMiB(indefiniteMemoryUsageBytes.unpadded);
    this.unpaddedHeapSizes =
        unpaddedHeapSizes.map(item => item + addendUnpadded);
  }

  /**
   * From a list of heap simulator traces, identify the one uses 0 as memory
   * space color.
   * @private
   */
  private getHbmHeapTraceByColor(
      color: number,
      traces?: proto.HeapSimulatorTrace[]): proto.HeapSimulatorTrace|null {
    if (!traces) {
      return null;
    }
    for (const trace of traces) {
      for (const event of trace.events || []) {
        if (!event.bufferId) continue;
        const buffer = this.idToBuffer[utils.toNumber(event.bufferId)];
        if (!buffer) continue;
        if (buffer.color !== color) break;
        return trace;
      }
    }
    return null;
  }

  /**
   * Creates a logical buffer id to buffer allocation map from
   * `bufferAllocations`.
   * @private
   */
  private initAllocations(bufferAllocations?: proto.BufferAllocationProto[]) {
    if (!bufferAllocations) {
      return;
    }
    for (const bufferAllocation of bufferAllocations) {
      const alloc = new BufferAllocation(bufferAllocation);
      for (const assigned of bufferAllocation.assigned || []) {
        if (!assigned.logicalBufferId) continue;
        this.idToBufferAllocation[utils.toNumber(assigned.logicalBufferId)] =
            alloc;
      }
    }
  }

  /**
   * Creates a sorted buffer list and an id to buffer map from
   * `logicalBuffers`.
   * @private
   */
  private initBuffers(logicalBuffers?: proto.LogicalBufferProto[]) {
    if (!logicalBuffers) {
      return;
    }
    for (const logicalBuffer of logicalBuffers) {
      if (!logicalBuffer.id) continue;
      const buffer = new LogicalBuffer(logicalBuffer);
      this.buffers.push(buffer);
      this.idToBuffer[buffer.id] = buffer;
      this.unSeenLogicalBuffers.add(buffer.id);
    }
  }

  /**
   * Constructs a mapping from name to HLO instruction.
   * @private
   */
  private initHloInstructions(hloModule?: proto.HloModuleProto) {
    if (!hloModule) {
      console.warn(
          'Missing hloModule, skipping unpadded allocation size analysis');
      return;
    }
    for (const comp of hloModule.computations || []) {
      for (const inst of comp.instructions || []) {
        if (!inst.name) continue;
        this.nameToHlo[inst.name] = new HloInstruction(inst);
      }
    }
  }

  /**
   * Accumulate data for use in a stacked bar plot.
   * We accumulate it in "program order" -- the order in which it was placed
   * into the logical_buffers sequence above was program order, and we iterate
   * that order to create data points.
   * @private
   */
  private initMaxHeap() {
    for (const id of this.peakLogicalBuffers) {
      const alloc = this.idToBufferAllocation[id];
      const groupName = alloc ? alloc.groupName : '';
      this.addHeapObject(this.idToBuffer[id], groupName);
    }
    if (this.rest !== 0) {
      const small =
          'small (<' + (this.smallBufferSize / 1024).toString() + ' KiB)';
      this.maxHeap.push({
        instructionName: small,
        sizeMiB: utils.bytesToMiB(this.rest),
        color: this.nColor++,
        groupName: small,
      });
    }
    const indexedMaxHeap = this.maxHeap.map((e, i) => {
      return {ind: i, val: e};
    });
    indexedMaxHeap.sort((a, b) => {
      const sizeA = a && a.val && a.val.sizeMiB ? a.val.sizeMiB : 0;
      const sizeB = b && b.val && b.val.sizeMiB ? b.val.sizeMiB : 0;
      return sizeB - sizeA;
    });
    this.maxHeapBySize = indexedMaxHeap.map(e => e.val);
    this.bySizeToMaxHeap = indexedMaxHeap.map(e => e.ind);
    this.maxHeapToBySize.length = this.maxHeap.length;
    for (let i = 0; i < this.bySizeToMaxHeap.length; i++) {
      this.maxHeapToBySize[this.bySizeToMaxHeap[i]] = i;
    }
  }

  /**
   * Initializes memory usage of the module.
   * @private
   */
  private initMemoryUsage(
      memorySpaceColor: number,
      bufferAssignment?: proto.BufferAssignmentProto) {
    if (!bufferAssignment) {
      console.warn('No buffer assignment info');
      return;
    }
    this.initBuffers(bufferAssignment.logicalBuffers);
    this.initAllocations(bufferAssignment.bufferAllocations);
    const trace = this.getHbmHeapTraceByColor(
        memorySpaceColor, bufferAssignment.heapSimulatorTraces);
    if (!trace) {
      console.warn('Missing hbm heap simulator trace.');
      return;
    }
    this.findPeakMemoryUsage(trace, memorySpaceColor);
  }

  /**
   * Creates a heap object that is displayed in a plot in the memory
   * visualization.
   * @private
   */
  private newHeapObject(
      buffer: LogicalBuffer, shape: Shape, inst: HloInstruction,
      groupName: string): HeapObject|null {
    if (!buffer || !inst) {
      return null;
    }
    const shapeIndex =
        buffer.shapeIndex.length ? ' {' + buffer.shapeIndex.join() + '}' : '';
    return {
      instructionName: buffer.instructionName + shapeIndex,
      logicalBufferId: buffer.id,
      unpaddedSizeMiB: shape ? utils.bytesToMiB(shape.unpaddedHeapSizeBytes()) :
                               0,
      tfOpName: inst.tfOpName,
      opcode: inst.opcode,
      sizeMiB: utils.bytesToMiB(buffer.size),
      color: this.nColor++,
      shape: shape ? shape.humanStringWithLayout() : '',
      groupName,
    };
  }
}
