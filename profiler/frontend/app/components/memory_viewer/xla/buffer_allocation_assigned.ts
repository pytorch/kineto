import * as proto from 'org_xprof/frontend/app/common/interfaces/hlo.proto';
import {toNumber} from 'org_xprof/frontend/app/common/utils/utils';

/**
 * HLO assigned buffer allocation representation.
 * @final
 */
export class BufferAllocationAssigned {
  logicalBufferId: number;
  offset: number;
  size: number;

  constructor(assigned?: proto.BufferAllocationProto.Assigned) {
    assigned = assigned || {};
    this.logicalBufferId = toNumber(assigned.logicalBufferId);
    this.offset = toNumber(assigned.offset);
    this.size = toNumber(assigned.size);
  }
}
