import {NgModule} from '@angular/core';
import {MemoryViewerMainModule} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer_main/memory_viewer_main_module';

import {MemoryViewer} from './memory_viewer';

/** A memory viewer module. */
@NgModule({
  declarations: [MemoryViewer],
  imports: [
    MemoryViewerMainModule,
  ],
  exports: [MemoryViewer]
})
export class MemoryViewerModule {
}
