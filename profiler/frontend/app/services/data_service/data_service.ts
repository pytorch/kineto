import {PlatformLocation} from '@angular/common';
import {HttpClient, HttpParams} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {API_PREFIX, CAPTURE_PROFILE_API, DATA_API, HOSTS_API, LOCAL_URL, PLUGIN_NAME, TOOLS_API} from 'org_xprof/frontend/app/common/constants/constants';
import {CaptureProfileOptions, CaptureProfileResponse} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Observable, of} from 'rxjs';
import {delay} from 'rxjs/operators';

import * as mockData from './mock_data';

/** Delay time for milisecond for testing */
const DELAY_TIME_MS = 1000;

/** The data service class that calls API and return response. */
@Injectable()
export class DataService {
  isLocalDevelopment = false;
  pathPrefix = '';

  constructor(
      private readonly httpClient: HttpClient, platformLocation: PlatformLocation) {
    this.isLocalDevelopment = platformLocation.pathname === LOCAL_URL;
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
  }

  captureProfile(options: CaptureProfileOptions):
      Observable<CaptureProfileResponse> {
    if (this.isLocalDevelopment) {
      return of({result: 'Done'});
    }
    const params =
        new HttpParams()
            .set('service_addr', options.serviceAddr)
            .set('is_tpu_name', options.isTpuName.toString())
            .set('duration', options.duration.toString())
            .set('num_retry', options.numRetry.toString())
            .set('worker_list', options.workerList)
            .set('host_tracer_level', options.hostTracerLevel.toString())
            .set('device_tracer_level', options.deviceTracerLevel.toString())
            .set('python_tracer_level', options.pythonTracerLevel.toString());
    return this.httpClient.get(this.pathPrefix + CAPTURE_PROFILE_API, {params});
  }

  getTools() {
    if (this.isLocalDevelopment) {
      return of(mockData.DATA_PLUGIN_PROFILE_TOOLS);
    }
    return this.httpClient.get(this.pathPrefix + TOOLS_API);
  }

  getHosts(run: string, tag: string) {
    if (this.isLocalDevelopment) {
      return of(mockData.DATA_PLUGIN_PROFILE_HOSTS).pipe(delay(DELAY_TIME_MS));
    }
    const params = new HttpParams().set('run', run).set('tag', tag);
    return this.httpClient.get(this.pathPrefix + HOSTS_API, {params});
  }

  getData(run: string, tag: string, host: string): Observable<DataTable> {
    if (this.isLocalDevelopment) {
      if (tag.startsWith('overview_page')) {
        return of(mockData.DATA_PLUGIN_PROFILE_OVERVIEW_PAGE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('input_pipeline_analyzer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_INPUT_PIPELINE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('tensorflow_stats')) {
        return of(mockData.DATA_PLUGIN_PROFILE_TENSORFLOW_STATS_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('memory_viewer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_MEMORY_VIEWER_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('op_profile')) {
        return of(mockData.DATA_PLUGIN_PROFILE_OP_PROFILE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('pod_viewer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_POD_VIEWER_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('kernel_stats')) {
        return of(mockData.DATA_PLUGIN_PROFILE_KERNEL_STATS_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('memory_profile')) {
        return of(mockData.DATA_PLUGIN_PROFILE_MEMORY_PROFILE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else {
        return of([]).pipe(delay(DELAY_TIME_MS));
      }
    }
    const params =
        new HttpParams().set('run', run).set('tag', tag).set('host', host);
    return this.httpClient.get(this.pathPrefix + DATA_API, {params}) as
        Observable<DataTable>;
  }

  exportDataAsCSV(run: string, tag: string, host: string) {
    const params = new HttpParams()
                       .set('run', run)
                       .set('tag', tag)
                       .set('host', host)
                       .set('tqx', 'out:csv;');
    window.open(this.pathPrefix + DATA_API + '?' + params.toString(), '_blank');
  }
}
