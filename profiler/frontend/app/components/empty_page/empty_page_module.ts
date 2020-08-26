import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';

import {CaptureProfileModule} from 'org_xprof/frontend/app/components/capture_profile/capture_profile_module';

import {EmptyPage} from './empty_page';

/** An empty page module. */
@NgModule({
  declarations: [EmptyPage],
  imports: [
    CommonModule,
    CaptureProfileModule,
  ],
  exports: [EmptyPage]
})
export class EmptyPageModule {
}
