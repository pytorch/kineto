import * as proto from 'org_xprof/frontend/app/common/interfaces/hlo.proto';
import {Shape} from './shape';

/**
 * HLO instructions are the IR used by the high-level XLA compiler.
 * @final
 */
export class HloInstruction {
  name: string;
  opcode: string;
  shape?: Shape;
  tfOpName?: string;

  constructor(instruction?: proto.HloInstructionProto) {
    instruction = instruction || {};
    this.name = instruction.name || '';
    this.opcode = instruction.opcode || '';
    if (instruction.shape) {
      this.shape = new Shape(instruction.shape);
    }
    if (instruction.metadata) {
      this.tfOpName = instruction.metadata.opName || '';
    }
  }
}
