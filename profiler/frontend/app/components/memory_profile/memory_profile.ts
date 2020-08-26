import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {MemoryProfileProtoOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {MemoryProfileBase} from 'org_xprof/frontend/app/components/memory_profile/memory_profile_base';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

/** A Memory Profile component. */
@Component({
  selector: 'memory-profile',
  templateUrl: './memory_profile.ng.html',
  styleUrls: ['./memory_profile.css']
})
export class MemoryProfile extends MemoryProfileBase {
  run = '';
  tag = '';
  host = '';

  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    super();
    route.params.subscribe(params => {
      this.update(params as NavigationEvent);
    });
  }

  update(event: NavigationEvent) {
    this.run = event.run || '';
    this.tag = event.tag || 'memory_profile';
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
          this.parseData(data as MemoryProfileProtoOrNull);
        });
  }
}
