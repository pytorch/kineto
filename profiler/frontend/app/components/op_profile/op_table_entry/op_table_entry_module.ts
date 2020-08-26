import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';

import {OpTableEntry} from './op_table_entry';

/** An op table entry view module. */
@NgModule({
  declarations: [OpTableEntry],
  imports: [CommonModule],
  exports: [OpTableEntry]
})
export class OpTableEntryModule {
}
