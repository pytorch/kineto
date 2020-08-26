import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {RunEnvironmentView} from './run_environment_view';

@NgModule({
  declarations: [RunEnvironmentView],
  imports: [MatCardModule],
  exports: [RunEnvironmentView]
})
export class RunEnvironmentViewModule {
}
