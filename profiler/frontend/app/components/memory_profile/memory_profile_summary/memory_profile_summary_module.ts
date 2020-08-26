import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MemoryProfileSummary} from './memory_profile_summary';

@NgModule({
  declarations: [MemoryProfileSummary],
  imports: [
    CommonModule,
    MatCardModule,
    MatTooltipModule,
  ],
  exports: [MemoryProfileSummary]
})
export class MemoryProfileSummaryModule {
}
