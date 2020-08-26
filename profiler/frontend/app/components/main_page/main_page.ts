import {Component, Input} from '@angular/core';
import {Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {Tool} from 'org_xprof/frontend/app/common/interfaces/tool';
import {getLoadingState} from 'org_xprof/frontend/app/store/selectors';
import {LoadingState} from 'org_xprof/frontend/app/store/state';

/** A main page component. */
@Component({
  selector: 'main-page',
  templateUrl: './main_page.ng.html',
  styleUrls: ['./main_page.scss']
})
export class MainPage {
  /** The tool datasets. */
  @Input() datasets: Tool[] = [];

  loading = true;
  loadingMessage = '';

  constructor(private readonly router: Router, store: Store<{}>) {
    store.select(getLoadingState).subscribe((loadingState: LoadingState) => {
      this.loading = loadingState.loading;
      this.loadingMessage = loadingState.message;
    });
  }

  updateTool(event: NavigationEvent) {
    this.loading = false;
    this.router.navigate([event.tag || 'empty', event]);
  }
}
