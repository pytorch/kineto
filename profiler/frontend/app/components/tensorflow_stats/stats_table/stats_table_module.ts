import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';

import {StatsTable} from './stats_table';

@NgModule({
  declarations: [StatsTable],
  imports: [
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
  ],
  exports: [StatsTable]
})
export class StatsTableModule {
}
