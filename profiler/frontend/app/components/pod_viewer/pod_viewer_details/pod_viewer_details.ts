import {Component, Input} from '@angular/core';
import {Store} from '@ngrx/store';

import {POD_STATS_RECORD_PROPERTY_MAP} from 'org_xprof/frontend/app/common/constants/constants';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord} from 'org_xprof/frontend/app/common/interfaces/data_table';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {getActivePodViewerInfoState} from 'org_xprof/frontend/app/store/selectors';

interface DetailInfo {
  title: string;
  value: string;
}

/** A pod viewer details view component. */
@Component({
  selector: 'pod-viewer-details',
  templateUrl: './pod_viewer_details.ng.html',
  styleUrls: ['./pod_viewer_details.scss']
})
export class PodViewerDetails {
  @Input() propertyMap = POD_STATS_RECORD_PROPERTY_MAP;

  info?: AllReduceOpInfo|ChannelInfo|PodStatsRecord;
  name = '';
  details: DetailInfo[] = [];
  hloNames = '';
  replicaGroups = '';
  description = '';

  constructor(private readonly store: Store<{}>) {
    this.store.select(getActivePodViewerInfoState)
        .subscribe((info: AllReduceOpInfo|ChannelInfo|PodStatsRecord|null) => {
          this.update(info);
        });
  }

  private updateSizeAndLatency(info: AllReduceOpInfo|ChannelInfo) {
    const dataSize = Number(info.dataSize || '0');
    const latency = Number(info.durationUs || '0');
    this.details.push({
      title: 'Data Transferred',
      value: utils.bytesToMiB(dataSize).toFixed(2) + ' MiB',
    });
    this.details.push({
      title: 'Latency',
      value: latency.toFixed(2) + ' Us',
    });
    this.details.push({
      title: 'BW',
      value: (latency !== 0 ? dataSize / latency / 1073.74 : 0).toFixed(2) +
          ' GiB/s',
    });
  }

  private updateAllReduceOpInfo(info: AllReduceOpInfo) {
    this.info = info;
    this.name = info.name || '';
    this.updateSizeAndLatency(info);
    (info.replicaGroups || []).forEach(replicaGroup => {
      if (replicaGroup.replicaIds && replicaGroup.replicaIds.length > 0) {
        this.replicaGroups += '{' + replicaGroup.replicaIds.join(',') + '} ';
      }
    });
    this.description = info.description || '';
  }

  private updateChannelInfo(info: ChannelInfo) {
    this.info = info;
    this.name = 'Channel # ' + (info.channelId || 0).toString();
    this.updateSizeAndLatency(info);
    this.details.push({
      title: 'Send Delay',
      value:
          utils.bytesToMiB(Number(info.sendDelayUs || '0')).toFixed(2) + ' Us',
    });
    (info.hloNames || []).forEach(hloName => {
      if (hloName) {
        this.hloNames += '"' + hloName + '" ';
      }
    });
    this.description = info.description || '';
  }

  private updatePodStatsRecord(info: PodStatsRecord) {
    this.info = info;
    this.name = 'Step breakdown of chip ' + (info.chipId || 0).toString();
    this.description = '';
    const total = info.totalDurationUs || 0;
    if (!total) {
      return;
    }
    this.propertyMap.forEach(metric => {
      const value = utils.getPodStatsRecordProperty(info, metric.key);
      this.details.push({
        title: metric.label,
        value: value.toFixed(2) + ' Us (' + (value / total).toFixed(2) + '%)',
      });
    });
  }

  update(info: AllReduceOpInfo|ChannelInfo|PodStatsRecord|null) {
    this.details = [];
    this.hloNames = '';
    this.replicaGroups = '';
    if (!info) {
      this.info = undefined;
      return;
    }
    if (info.hasOwnProperty('channelId')) {
      this.updateChannelInfo(info as ChannelInfo);
    } else if (info.hasOwnProperty('chipId')) {
      this.updatePodStatsRecord(info as PodStatsRecord);
    } else {
      this.updateAllReduceOpInfo(info as AllReduceOpInfo);
    }
  }
}
