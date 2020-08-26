import {Component} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {MatSnackBar} from '@angular/material/snack-bar';
import {Store} from '@ngrx/store';
import {CaptureProfileOptions, CaptureProfileResponse} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setCapturingProfileAction} from 'org_xprof/frontend/app/store/actions';
import {getCapturingProfileState} from 'org_xprof/frontend/app/store/selectors';
import {Observable} from 'rxjs';

import {CaptureProfileDialog} from './capture_profile_dialog/capture_profile_dialog';

const DELAY_TIME_MS = 1000;

/** A capture profile view component. */
@Component({
  selector: 'capture-profile',
  templateUrl: './capture_profile.ng.html',
  styleUrls: ['./capture_profile.scss']
})
export class CaptureProfile {
  captureButtonLabel = 'Capture Profile';
  capturingProfile: Observable<boolean>;

  constructor(
      private readonly dialog: MatDialog,
      private readonly snackBar: MatSnackBar,
      private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    this.capturingProfile = this.store.select(getCapturingProfileState);
  }

  private openSnackBar(message: string) {
    this.snackBar.open(message, 'Close now!', {duration: 5000});
  }

  openDialog() {
    this.dialog.open(CaptureProfileDialog).afterClosed().subscribe(options => {
      if (!options) {
        return;
      }

      this.store.dispatch(setCapturingProfileAction({capturingProfile: true}));
      this.dataService.captureProfile(options as CaptureProfileOptions)
          .subscribe(
              (response: CaptureProfileResponse) => {
                this.store.dispatch(
                    setCapturingProfileAction({capturingProfile: false}));
                if (!response) {
                  return;
                }
                if (response.error) {
                  this.openSnackBar(
                      'Failed to capture profile: ' + response.error);
                  return;
                }
                if (response.result) {
                  this.openSnackBar(response.result);
                  setTimeout(() => {
                    document.dispatchEvent(new Event('plugin-reload'));
                  }, DELAY_TIME_MS);
                }
              },
              error => {
                this.store.dispatch(
                    setCapturingProfileAction({capturingProfile: false}));
                const errorMessage: string =
                    error && error.toString() ? error.toString() : '';
                this.openSnackBar('Failed to capture profile: ' + errorMessage);
              });
    });
  }
}
