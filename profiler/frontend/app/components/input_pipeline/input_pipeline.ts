import {InputPipelineDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Component} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {InputPipelineCommon} from './input_pipeline_common';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';

/** An input pipeline component. */
@Component({
  selector: 'input-pipeline',
  templateUrl: './input_pipeline.ng.html',
  styleUrls: ['./input_pipeline.css']
})
export class InputPipeline extends InputPipelineCommon {
  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    super();
    route.params.subscribe(params => {
      this.update(params as NavigationEvent);
    });
  }

  update(event: NavigationEvent) {
    const run = event.run || '';
    const tag = event.tag || 'input_pipeline_analyzer';
    const host = event.host || '';

    setLoadingState(true, this.store);

    this.dataService.getData(run, tag, host)
        .subscribe(data => {
          setLoadingState(false, this.store);
          data = (data || []) as InputPipelineDataTable[];
          this.parseCommonInputData(data);
        });
  }
}
