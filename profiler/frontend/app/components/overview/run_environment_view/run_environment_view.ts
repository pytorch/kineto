import {Component, Input} from '@angular/core';

import {RunEnvironment} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A run environment view component. */
@Component({
  selector: 'run-environment-view',
  templateUrl: './run_environment_view.ng.html',
  styleUrls: ['./run_environment_view.scss']
})
export class RunEnvironmentView {
  /** The run environment data. */
  @Input()
  set runEnvironment(data: RunEnvironment|null) {
    data = data || {};
    data.p = data.p || {};
    this.deviceCoreCount = data.p.device_core_count || '';
    this.deviceType = data.p.device_type || '';
    this.hostCount = data.p.host_count || '';
  }

  title = 'Run Environment';
  deviceCoreCount = '';
  deviceType = '';
  hostCount = '';
}
