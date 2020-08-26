import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBarModule} from '@angular/material/snack-bar';

import {CaptureProfile} from './capture_profile';
import {CaptureProfileDialog} from './capture_profile_dialog/capture_profile_dialog';
import {CaptureProfileDialogModule} from './capture_profile_dialog/capture_profile_dialog_module';

/** A capture profile view module. */
@NgModule({
  declarations: [CaptureProfile],
  imports: [
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    MatProgressSpinnerModule,
    CaptureProfileDialogModule,
    MatSnackBarModule,
  ],
  exports: [CaptureProfile],
  entryComponents: [CaptureProfileDialog],
})
export class CaptureProfileModule {
}
