import {NgModule} from '@angular/core';
import {MatDividerModule} from '@angular/material/divider';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {AnalysisSummaryModule} from 'org_xprof/frontend/app/components/input_pipeline/analysis_summary/analysis_summary_module';
import {DeviceSideAnalysisDetailModule} from 'org_xprof/frontend/app/components/input_pipeline/device_side_analysis_detail/device_side_analysis_detail_module';
import {HostSideAnalysisDetailModule} from 'org_xprof/frontend/app/components/input_pipeline/host_side_analysis_detail/host_side_analysis_detail_module';

import {InputPipeline} from './input_pipeline';

@NgModule({
  declarations: [InputPipeline],
  imports: [
    AnalysisSummaryModule,
    DeviceSideAnalysisDetailModule,
    DiagnosticsViewModule,
    HostSideAnalysisDetailModule,
    MatDividerModule,
  ],
  exports: [InputPipeline]
})
export class InputPipelineModule {
}
