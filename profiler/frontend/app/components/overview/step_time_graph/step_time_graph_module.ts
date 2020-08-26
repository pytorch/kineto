import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {StepTimeGraph} from './step_time_graph';

@NgModule({
  declarations: [
    StepTimeGraph
  ],
  imports: [
    MatCardModule
  ],
  exports: [StepTimeGraph]
})
export class StepTimeGraphModule { }
