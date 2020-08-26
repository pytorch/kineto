import {NgModule} from '@angular/core';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSidenavModule} from '@angular/material/sidenav';
import {RouterModule, Routes} from '@angular/router';
import {EmptyPage} from 'org_xprof/frontend/app/components/empty_page/empty_page';
import {EmptyPageModule} from 'org_xprof/frontend/app/components/empty_page/empty_page_module';
import {InputPipeline} from 'org_xprof/frontend/app/components/input_pipeline/input_pipeline';
import {InputPipelineModule} from 'org_xprof/frontend/app/components/input_pipeline/input_pipeline_module';
import {KernelStats} from 'org_xprof/frontend/app/components/kernel_stats/kernel_stats';
import {KernelStatsModule} from 'org_xprof/frontend/app/components/kernel_stats/kernel_stats_module';
import {MemoryProfile} from 'org_xprof/frontend/app/components/memory_profile/memory_profile';
import {MemoryProfileModule} from 'org_xprof/frontend/app/components/memory_profile/memory_profile_module';
import {MemoryViewer} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer';
import {MemoryViewerModule} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer_module';
import {OpProfile} from 'org_xprof/frontend/app/components/op_profile/op_profile';
import {OpProfileModule} from 'org_xprof/frontend/app/components/op_profile/op_profile_module';
import {Overview} from 'org_xprof/frontend/app/components/overview/overview';
import {OverviewModule} from 'org_xprof/frontend/app/components/overview/overview_module';
import {PodViewer} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer';
import {PodViewerModule} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer_module';
import {SideNavModule} from 'org_xprof/frontend/app/components/sidenav/sidenav_module';
import {TensorflowStats} from 'org_xprof/frontend/app/components/tensorflow_stats/tensorflow_stats';
import {TensorflowStatsModule} from 'org_xprof/frontend/app/components/tensorflow_stats/tensorflow_stats_module';
import {TraceViewer} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer';
import {TraceViewerModule} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer_module';

import {MainPage} from './main_page';

/** The list of all routes available in the application. */
export const routes: Routes = [
  {path: 'empty', component: EmptyPage},
  {path: 'overview_page', component: Overview},
  {path: 'overview_page@', component: Overview},
  {path: 'overview_page^', component: Overview},
  {path: 'input_pipeline_analyzer', component: InputPipeline},
  {path: 'input_pipeline_analyzer@', component: InputPipeline},
  {path: 'input_pipeline_analyzer^', component: InputPipeline},
  {path: 'kernel_stats', component: KernelStats},
  {path: 'kernel_stats^', component: KernelStats},
  {path: 'memory_profile#', component: MemoryProfile},
  {path: 'memory_profile^', component: MemoryProfile},
  {path: 'memory_viewer', component: MemoryViewer},
  {path: 'op_profile', component: OpProfile},
  {path: 'pod_viewer', component: PodViewer},
  {path: 'tensorflow_stats', component: TensorflowStats},
  {path: 'tensorflow_stats^', component: TensorflowStats},
  {path: 'trace_viewer', component: TraceViewer},
  {path: 'trace_viewer#', component: TraceViewer},
  {path: 'trace_viewer@', component: TraceViewer},
  {path: 'trace_viewer^', component: TraceViewer},
  {path: '**', component: EmptyPage},
];

/** A main page module. */
@NgModule({
  declarations: [MainPage],
  imports: [
    MatProgressBarModule,
    MatSidenavModule,
    EmptyPageModule,
    SideNavModule,
    TraceViewerModule,
    OverviewModule,
    InputPipelineModule,
    KernelStatsModule,
    MemoryProfileModule,
    MemoryViewerModule,
    OpProfileModule,
    PodViewerModule,
    TensorflowStatsModule,
    RouterModule.forRoot(routes),
  ],
  exports: [MainPage]
})
export class MainPageModule {
}
