import {CommonModule} from '@angular/common'
import {NgModule} from '@angular/core';
import {SafePipe} from './safe_pipe';

@NgModule(
    {imports: [CommonModule], declarations: [SafePipe], exports: [SafePipe]})
export class PipesModule {
}
