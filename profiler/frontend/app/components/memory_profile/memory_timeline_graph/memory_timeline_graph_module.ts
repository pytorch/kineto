import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {MemoryTimelineGraph} from './memory_timeline_graph';

@NgModule({
  declarations: [MemoryTimelineGraph],
  imports: [MatCardModule],
  exports: [MemoryTimelineGraph]
})
export class MemoryTimelineGraphModule {
}
