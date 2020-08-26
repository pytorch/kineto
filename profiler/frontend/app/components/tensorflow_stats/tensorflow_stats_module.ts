import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {FlopRateChartModule} from 'org_xprof/frontend/app/components/tensorflow_stats/flop_rate_chart/flop_rate_chart_module';
import {ModelPropertiesModule} from 'org_xprof/frontend/app/components/tensorflow_stats/model_properties/model_properties_module';
import {OperationsTableModule} from 'org_xprof/frontend/app/components/tensorflow_stats/operations_table/operations_table_module';
import {SelfTimeChartModule} from 'org_xprof/frontend/app/components/tensorflow_stats/self_time_chart/self_time_chart_module';
import {StatsTableModule} from 'org_xprof/frontend/app/components/tensorflow_stats/stats_table/stats_table_module';

import {TensorflowStats} from './tensorflow_stats';

/** An op profile module. */
@NgModule({
  declarations: [TensorflowStats],
  imports: [
    CommonModule,
    MatButtonModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    FlopRateChartModule,
    ModelPropertiesModule,
    OperationsTableModule,
    SelfTimeChartModule,
    StatsTableModule,
  ],
  exports: [TensorflowStats]
})
export class TensorflowStatsModule {
}
