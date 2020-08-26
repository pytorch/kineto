import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {DiagnosticsView} from './diagnostics_view';

@NgModule({
  declarations: [DiagnosticsView],
  imports: [CommonModule],
  exports: [DiagnosticsView]
})
export class DiagnosticsViewModule {
}
