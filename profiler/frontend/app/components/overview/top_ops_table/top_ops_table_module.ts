import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';

import {TopOpsTable} from './top_ops_table';

@NgModule({
  declarations: [TopOpsTable],
  imports: [MatExpansionModule],
  exports: [TopOpsTable]
})
export class TopOpsTableModule {
}
