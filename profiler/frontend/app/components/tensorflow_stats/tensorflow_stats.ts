import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {IdleOption, OpExecutor, OpKind} from 'org_xprof/frontend/app/common/constants/enums';
import {TensorflowStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

/** A TensorFlow Stats component. */
@Component({
  selector: 'tensorflow-stats',
  templateUrl: './tensorflow_stats.ng.html',
  styleUrls: ['./tensorflow_stats.css']
})
export class TensorflowStats {
  data: TensorflowStatsData[]|null = null;
  selectedData: TensorflowStatsData|null = null;
  run = '';
  tag = '';
  host = '';
  idleMenuButtonLabel = IdleOption.NO;
  idleOptionItems = [IdleOption.YES, IdleOption.NO];
  opExecutorDevice = OpExecutor.DEVICE;
  opExecutorHost = OpExecutor.HOST;
  opKindName = OpKind.NAME;
  opKindType = OpKind.TYPE;
  architecture = '';
  task = '';
  hasDataRow = true;
  hasDeviceData = false;

  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    route.params.subscribe(params => {
      this.idleMenuButtonLabel = IdleOption.NO;
      this.update(params as NavigationEvent);
    });
  }

  exportDataAsCSV() {
    this.dataService.exportDataAsCSV(this.run, this.tag, this.host);
  }

  setIdleOption(option: IdleOption = IdleOption.NO) {
    this.idleMenuButtonLabel = option;
    if (!this.data) {
      this.selectedData = null;
      return;
    }

    if (option === IdleOption.YES) {
      this.selectedData = this.data[0] || null;
    } else {
      this.selectedData = this.data[1] || null;
    }
    if (this.selectedData && this.selectedData.p) {
      this.architecture = this.selectedData.p.architecture_type || '';
      this.task = this.selectedData.p.task_type || '';
    }
    this.hasDeviceData = false;
    if (this.selectedData && this.selectedData.rows) {
      this.hasDeviceData = !!this.selectedData.rows.find(row => {
        return row && row.c && row.c[1] && row.c[1].v === OpExecutor.DEVICE;
      });
    }
  }

  update(event: NavigationEvent) {
    this.run = event.run || '';
    this.tag = event.tag || 'tensorflow_stats';
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

          this.data = data as TensorflowStatsData[] || [];
          this.setIdleOption();
        });
  }

  changeHasDataRows(event: boolean) {
    this.hasDataRow = event;
  }
}
