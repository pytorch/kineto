import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';

import {MemoryBreakdownTable} from './memory_breakdown_table';

@NgModule({
  declarations: [MemoryBreakdownTable],
  imports: [
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
  ],
  exports: [MemoryBreakdownTable]
})
export class MemoryBreakdownTableModule {
}
