import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatSliderModule} from '@angular/material/slider';

import {KernelStatsChart} from './kernel_stats_chart';

@NgModule({
  declarations: [KernelStatsChart],
  imports: [
    MatFormFieldModule,
    MatInputModule,
    MatSliderModule,
  ],
  exports: [KernelStatsChart]
})
export class KernelStatsChartModule {
}
