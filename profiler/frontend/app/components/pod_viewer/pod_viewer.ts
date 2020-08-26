import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {PodViewerDatabaseOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setActivePodViewerInfoAction, setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

import {PodViewerCommon} from './pod_viewer_common';

/** A pod viewer component. */
@Component({
  selector: 'pod-viewer',
  templateUrl: './pod_viewer.ng.html',
  styleUrls: ['./pod_viewer.css']
})
export class PodViewer extends PodViewerCommon {
  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      readonly store: Store<{}>) {
    super(store);
    route.params.subscribe(params => {
      this.update(params as NavigationEvent);
    });
  }

  selectedAllReduceOpChart(allReduceOpIndex: number) {
    this.store.dispatch(setActivePodViewerInfoAction({
      activePodViewerInfo:
          this.allReduceOpDb ? this.allReduceOpDb[allReduceOpIndex] : null
    }));
  }

  update(event: NavigationEvent) {
    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading data',
      }
    }));

    this.dataService
        .getData(event.run || '', event.tag || 'pod_viewer', event.host || '')
        .subscribe(data => {
          this.store.dispatch(setLoadingStateAction({
            loadingState: {
              loading: false,
              message: '',
            }
          }));

          this.parseData(data as PodViewerDatabaseOrNull);
        });
  }
}
