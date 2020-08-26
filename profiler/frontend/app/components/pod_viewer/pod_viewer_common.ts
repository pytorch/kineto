import {Store} from '@ngrx/store';
import {POD_STATS_RECORD_PROPERTY_MAP} from 'org_xprof/frontend/app/common/constants/constants';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord, PodViewerDatabaseOrNull, PodViewerRunEnvironment, PrimitiveTypeNumberStringOrUndefined} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {setActivePodViewerInfoAction} from 'org_xprof/frontend/app/store/actions';

/** A common class of pod viewer component. */
export class PodViewerCommon {
  data: PodViewerDatabaseOrNull = null;
  minStep: number = 0;
  maxStep: number = 0;
  selectedStep: string = '';
  allReduceOpDb?: AllReduceOpInfo[];
  allReduceOpChartData?: PrimitiveTypeNumberStringOrUndefined[][];
  channelDb?: ChannelInfo[];
  channelDbForChart?: ChannelInfo[];
  channelChartData?: PrimitiveTypeNumberStringOrUndefined[][];
  coreIdToReplicaIdMap?: {[key: /* uint32 */ string]: /* uint32 */ number};
  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  podStatsPerCore?: {[key: string]: PodStatsRecord};
  podStatsForChart?: PodStatsRecord[];
  podStatsChartData?: PrimitiveTypeNumberStringOrUndefined[][];
  podStatsRecordPropertyMap: Array<{key: string, label: string}> =
      POD_STATS_RECORD_PROPERTY_MAP;
  runEnvironment?: PodViewerRunEnvironment;

  constructor(readonly store: Store<{}>) {}

  private updateSteps() {
    this.minStep = 0;
    this.maxStep = 0;
    if (!this.data || !this.data.podStatsSequence ||
        !this.data.podStatsSequence.podStatsMap) {
      this.updateSelectedStep(this.minStep);
      return;
    }
    const podStats = this.data.podStatsSequence.podStatsMap[0];
    if (!podStats) {
      this.updateSelectedStep(this.minStep);
      return;
    }
    this.minStep = podStats.stepNum || 0;
    this.maxStep = Math.max(
        this.minStep,
        this.minStep + this.data.podStatsSequence.podStatsMap.length - 1);
    this.updateSelectedStep(this.minStep);
  }

  updateSelectedStep(step: number) {
    if (!this.data || !this.data.podStatsSequence ||
        !this.data.podStatsSequence.podStatsMap) {
      this.selectedStep = '';
      this.channelDb = undefined;
      this.podStatsPerCore = undefined;
      return;
    }

    const podStats =
        this.data.podStatsSequence.podStatsMap[step - this.minStep];
    if (!podStats) {
      this.selectedStep = '';
      this.channelDb = undefined;
      this.podStatsPerCore = undefined;
      return;
    }
    // Negative step number indicates incomplete step.
    this.selectedStep = step >> 0 > 0 ? step.toString() : 'incomplete step';
    this.allReduceOpDb =
        (podStats.allReduceOpDb || [])
            .slice(0)
            .sort(
                (a, b) => ((a.durationUs || 0) < (b.durationUs || 0)) ? 1 : -1);
    this.allReduceOpChartData = this.allReduceOpDb.map(allReduceOpInfo => {
      let name = allReduceOpInfo.name || '';
      name = name.replace(/ll-reduce.|usion.|ll-reduce|usion/, '');
      name = name.replace(/all-to-all.|all-to-all/, 'l');
      name = name.length > 1 ? name : name + '0';
      return [name, allReduceOpInfo.durationUs];
    });
    this.allReduceOpChartData.unshift(['name', 'Duration (us)']);
    this.channelDb =
        (podStats.channelDb || [])
            .sort(
                (a, b) =>
                    ((a.channelId || '0') > (b.channelId || '0')) ? 1 : -1);
    this.channelDbForChart =
        (podStats.channelDb || [])
            .slice(0)
            .sort(
                (a, b) => ((a.durationUs || 0) < (b.durationUs || 0)) ? 1 : -1);
    this.channelChartData = this.channelDbForChart.map(
        channelInfo =>
            [channelInfo.channelId,
             channelInfo.durationUs,
    ]);
    this.channelChartData.unshift(['channelId', 'Duration (us)']);
    this.coreIdToReplicaIdMap = podStats.coreIdToReplicaIdMap || {};
    this.podStatsPerCore = podStats.podStatsPerCore;
    if (this.podStatsPerCore) {
      Object.values(this.podStatsPerCore).forEach(podStatsRecord => {
        let lowFlopsComputeUs = podStatsRecord.totalDurationUs || 0;
        this.podStatsRecordPropertyMap.forEach(propertyMap => {
          if (propertyMap.key === 'lowFlopsComputeUs') {
            return;
          }
          lowFlopsComputeUs -=
              utils.getPodStatsRecordProperty(podStatsRecord, propertyMap.key);
        });
        podStatsRecord.lowFlopsComputeUs = lowFlopsComputeUs;
      });
    }
    this.podStatsForChart = Object.values(this.podStatsPerCore || {})
                                .map(podStatsRecord => podStatsRecord);
    this.podStatsForChart.sort((a, b) => {
      if (a.chipId === b.chipId) {
        return ((a.nodeId || 0) > (b.nodeId || 0)) ? 1 : -1;
      }
      return ((a.chipId || 0) > (b.chipId || 0)) ? 1 : -1;
    });
    this.podStatsChartData =
        this.podStatsForChart.map(this.parsePodStatsRecord);
    const metrics = this.podStatsRecordPropertyMap.map(metric => metric.label);
    metrics.unshift('metrics');
    this.podStatsChartData.unshift(metrics);
  }

  parsePodStatsRecord(podStatsRecord: PodStatsRecord): Array<string|number> {
    return [
      '(' + (podStatsRecord.chipId || 0).toString() + ',' +
          (podStatsRecord.nodeId || 0).toString() + ')',
      podStatsRecord.highFlopsComputeUs || 0,
      podStatsRecord.lowFlopsComputeUs || 0,
      podStatsRecord.hostInfeedDurationUs || 0,
      podStatsRecord.hostOutfeedDurationUs || 0,
      podStatsRecord.allReduceComputeDurationUs || 0,
      podStatsRecord.allReduceSyncDurationUs || 0,
      podStatsRecord.sendDurationUs || 0,
      podStatsRecord.recvDurationUs || 0,
    ];
  }

  selectedAllReduceOpChart(allReduceOpIndex: number) {
    this.store.dispatch(setActivePodViewerInfoAction({
      activePodViewerInfo:
          this.allReduceOpDb ? this.allReduceOpDb[allReduceOpIndex] : null
    }));
  }

  selectedChannelDb(channelDbIndex: number) {
    this.store.dispatch(setActivePodViewerInfoAction({
      activePodViewerInfo: this.channelDb ? this.channelDb[channelDbIndex] :
                                            null
    }));
  }

  selectedChannelChart(channelIndex: number) {
    this.store.dispatch(setActivePodViewerInfoAction({
      activePodViewerInfo:
          this.channelDbForChart ? this.channelDbForChart[channelIndex] : null
    }));
  }

  selectedPodStatsChart(podStatsIndex: number) {
    this.store.dispatch(setActivePodViewerInfoAction({
      activePodViewerInfo:
          this.podStatsForChart ? this.podStatsForChart[podStatsIndex] : null
    }));
  }

  setDiagnostics(data: PodViewerDatabaseOrNull) {
    if (!data || !data.diagnostics) return;
    this.diagnostics.info = data.diagnostics.info || [];
    this.diagnostics.warnings = data.diagnostics.warnings || [];
    this.diagnostics.errors = data.diagnostics.errors || [];
  }

  parseData(data: PodViewerDatabaseOrNull) {
    this.data = data;
    this.setDiagnostics(this.data);
    this.updateSteps();
    this.runEnvironment = this.data ? this.data.runEnvironment : undefined;
  }
}
