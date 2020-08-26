import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';
import {MemoryBreakdownTableModule} from 'org_xprof/frontend/app/components/memory_profile/memory_breakdown_table/memory_breakdown_table_module';
import {MemoryProfileSummaryModule} from 'org_xprof/frontend/app/components/memory_profile/memory_profile_summary/memory_profile_summary_module';
import {MemoryTimelineGraphModule} from 'org_xprof/frontend/app/components/memory_profile/memory_timeline_graph/memory_timeline_graph_module';

import {MemoryProfile} from './memory_profile';

/** A memory profile module. */
@NgModule({
  declarations: [MemoryProfile],
  imports: [
    CommonModule,
    MatDividerModule,
    MatIconModule,
    MatSelectModule,
    MatOptionModule,
    MemoryProfileSummaryModule,
    MemoryTimelineGraphModule,
    MemoryBreakdownTableModule,
  ],
  exports: [MemoryProfile]
})
export class MemoryProfileModule {
}
