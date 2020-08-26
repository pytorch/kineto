import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

/** A Kernel Stats component. */
@Component({
  selector: 'kernel-stats',
  templateUrl: './kernel_stats.ng.html',
  styleUrls: ['./kernel_stats.css']
})
export class KernelStats {
  data: SimpleDataTable|null = null;
  run = '';
  tag = '';
  host = '';
  hasDataRow = false;

  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    route.params.subscribe(params => {
      this.update(params as NavigationEvent);
    });
  }

  exportDataAsCSV() {
    this.dataService.exportDataAsCSV(this.run, this.tag, this.host);
  }

  update(event: NavigationEvent) {
    this.run = event.run || '';
    this.tag = event.tag || 'kernel_stats';
    this.host = event.host || '';

    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading data',
      }
    }));

    this.dataService.getData(this.run, this.tag, this.host)
        .subscribe(data => {
          this.store.dispatch(setLoadingStateAction({
            loadingState: {
              loading: false,
              message: '',
            }
          }));

          this.data = (data as SimpleDataTable[] || [{}])[0];
          this.hasDataRow = !!(this.data.rows) && this.data.rows.length > 0;
        });
  }
}
