import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';

import {HostSideAnalysisDetail} from './host_side_analysis_detail';

@NgModule({
  declarations: [HostSideAnalysisDetail],
  imports: [
    CommonModule,
    MatExpansionModule,
  ],
  exports: [HostSideAnalysisDetail]
})
export class HostSideAnalysisDetailModule {
}
