import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {OpDetails} from './op_details';

/** An op details view module. */
@NgModule({
  declarations: [OpDetails],
  imports: [CommonModule, MatCardModule],
  exports: [OpDetails]
})
export class OpDetailsModule {
}
