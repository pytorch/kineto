import {Component, Input} from '@angular/core';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss']
})
export class DiagnosticsView {
  /** Error and warning messages for diagnosing profiling issues */
  @Input() diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
}
