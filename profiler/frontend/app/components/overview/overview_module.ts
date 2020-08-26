import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {NormalizedAcceleratorPerformanceViewModule} from 'org_xprof/frontend/app/components/overview/normalized_accelerator_performance_view/normalized_accelerator_performance_view_module';
import {PerformanceSummaryModule} from 'org_xprof/frontend/app/components/overview/performance_summary/performance_summary_module';
import {RecommendationResultViewModule} from 'org_xprof/frontend/app/components/overview/recommendation_result_view/recommendation_result_view_module';
import {RunEnvironmentViewModule} from 'org_xprof/frontend/app/components/overview/run_environment_view/run_environment_view_module';
import {StepTimeGraphModule} from 'org_xprof/frontend/app/components/overview/step_time_graph/step_time_graph_module';
import {TopOpsTableModule} from 'org_xprof/frontend/app/components/overview/top_ops_table/top_ops_table_module';

import {Overview} from './overview';

/** An overview page module. */
@NgModule({
  declarations: [Overview],
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    PerformanceSummaryModule,
    RecommendationResultViewModule,
    RunEnvironmentViewModule,
    StepTimeGraphModule,
    TopOpsTableModule,
    NormalizedAcceleratorPerformanceViewModule,
  ],
  exports: [Overview]
})
export class OverviewModule {
}
