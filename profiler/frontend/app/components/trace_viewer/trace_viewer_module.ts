import {NgModule} from '@angular/core';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer],
  imports: [PipesModule],
  exports: [TraceViewer]
})
export class TraceViewerModule {
}
