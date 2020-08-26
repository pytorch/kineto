import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

/** An op profile data class. */
export class OpProfileData {
  bwColor?: string;
  flopsColor?: string;
  memoryUtilizationPercent?: string;
  utilizationPercent?: string;

  update(node?: Node) {
    if (node) {
      let utilization = utils.utilization(node);
      this.flopsColor = utils.flopsColor(utilization);
      this.utilizationPercent = utils.percent(utilization);

      utilization = utils.memoryUtilization(node);
      this.bwColor = utils.bwColor(utilization);
      this.memoryUtilizationPercent = utils.percent(utilization);
    } else {
      this.bwColor = undefined;
      this.flopsColor = undefined;
      this.memoryUtilizationPercent = undefined;
      this.utilizationPercent = undefined;
    }
  }
}
