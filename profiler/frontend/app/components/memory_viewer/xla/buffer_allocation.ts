import * as proto from 'org_xprof/frontend/app/common/interfaces/hlo.proto';
import {toNumber} from 'org_xprof/frontend/app/common/utils/utils';

import {BufferAllocationAssigned} from './buffer_allocation_assigned';

/**
 * HLO buffer allocation representation.
 * @final
 */
export class BufferAllocation {
  index: number;
  size: number;
  isThreadLocal: boolean;
  assigned: BufferAllocationAssigned[];
  groupName: string;

  constructor(allocation?: proto.BufferAllocationProto) {
    allocation = allocation || {};
    this.index = toNumber(allocation.index);
    this.size = toNumber(allocation.size);
    this.isThreadLocal = allocation.isThreadLocal || false;
    this.assigned = this.getAllocation(allocation);
    this.groupName = this.getGroupName(allocation);
  }

  private getAllocation(allocation: proto.BufferAllocationProto) {
    return allocation.assigned ?
        allocation.assigned.map(
            assigned => new BufferAllocationAssigned(assigned)) :
        [];
  }

  private getGroupName(allocation: proto.BufferAllocationProto) {
    if (allocation.isEntryComputationParameter) {
      return 'Parameter';
    } else if (allocation.maybeLiveOut) {
      return 'Output';
    } else if (allocation.isThreadLocal) {
      return 'Thread-local';
    }
    return 'Temporary';
  }
}
