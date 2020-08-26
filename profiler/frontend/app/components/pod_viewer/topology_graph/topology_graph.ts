import {Component, ElementRef, EventEmitter, Input, OnChanges, Output, SimpleChanges} from '@angular/core';
import {Store} from '@ngrx/store';
import {TpuClass} from 'org_xprof/frontend/app/common/constants/enums';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord, PodViewerRunEnvironment} from 'org_xprof/frontend/app/common/interfaces/data_table';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {getActivePodViewerInfoState} from 'org_xprof/frontend/app/store/selectors';

interface ColorInfo {
  color: string;
  label: string;
}

interface ElementInfo {
  id?: string;
  rid?: number;
  x: number;
  y: number;
}

interface ArrowElementInfo extends ElementInfo {
  rotate?: number;
  scale?: number;
}

const BORDER_WIDTH = 1;
const CONTAINER_MARGIN = 4;
const CHIP_PADDING = 4;
const HOST_MARGIN = 5;
const HOST_PADDING = 5;
const LABEL_HEIGHT = 14;
const LABEL_PADDING = 5;
const LABEL_WIDTH = 20;
const NODE_HEIGHT = 30;
const NODE_WIDTH = 15;
const TOOLTIP_OFFSET_X = 20;
const TOOLTIP_OFFSET_Y = 35;
const NODE_COLORS = [
  '#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8',
  '#253494', '#081d58'
];
const KELLY_COLORS = [
  '#ffb300',  // Vivid yellow
  '#803e75',  // Strong purple
  '#ff6800',  // Vivid orange
  '#a6bdd7',  // Very light blue
  '#c10020',  // Vivid red
  '#cea262',  // Grayish yellow
  '#817066',  // Medium gray
  '#007d34',  // Vivid green
  '#f6768e',  // Strong purplish pink
  '#00538a',  // Strong blue
  '#ff7a5c',  // Strong yellowish pink
  '#53377a',  // Strong violet
  '#ff8e00',  // Vivid orange yellow
  '#b32851',  // Strong purplish red
  '#f4c800',  // Vivid greenish yellow
  '#7f180d',  // Strong reddish brown
  '#93aa00',  // Vivid yellowish green
  '#593315',  // Deep yellowish brown
  '#f13a13',  // Vivid reddish orange
  '#232c16',  // Dark olive green
];

/** A topology graph view component. */
@Component({
  selector: 'topology-graph',
  templateUrl: './topology_graph.ng.html',
  styleUrls: ['./topology_graph.scss']
})
export class TopologyGraph implements OnChanges {
  /** The channel dababase. */
  @Input() channelDb?: ChannelInfo[];

  /** The replica id map with core id as key. */
  @Input()
  coreIdToReplicaIdMap?: {[key: /* uint32 */ string]: /* uint32 */ number};

  /** The metric map. */
  @Input() metricMap?: Array<{key: string, label: string}>;

  /** The pod stats per core. */
  @Input() podStatsPerCore?: {[key: string]: PodStatsRecord};

  /** The run environment of a pod viewer. */
  @Input() runEnvironment?: PodViewerRunEnvironment;

  @Input() classifyTpuFunc: (tpuType: string) => TpuClass = this.getTpuClass;

  /** The event when the selection of the channel is changed. */
  @Output() selected = new EventEmitter<number>();

  hosts: ElementInfo[] = [];
  labels: ElementInfo[] = [];
  nodes: ElementInfo[] = [];
  colorInfos: ColorInfo[] = [];
  channels: number[] = [];
  arrows: ArrowElementInfo[] = [];
  tpuType = '';
  xDimension = 0;
  yDimension = 0;
  containerHeight = 0;
  containerWidth = 0;
  hostColumns = 0;
  hostRows = 0;
  hostHeight = 0;
  hostXStride = 1;
  hostYStride = 1;
  hostWidth = 0;
  nodesPerChip = 2;
  selectedMetric = '';
  selectedMetricLabel = '';
  channelCount = 0;
  firstChannel = 0;
  lastChannel = 0;
  selectedChannelIndex = 0;
  selectedChannelId = 0;
  tooltipText = '';
  tooltipX = 0;
  tooltipY = 0;
  info?: AllReduceOpInfo|ChannelInfo|PodStatsRecord;

  constructor(
      private readonly elRef: ElementRef, private readonly store: Store<{}>) {
    this.createColorInfos();
    this.store.select(getActivePodViewerInfoState).subscribe((info) => {
      this.updateReplicaGroupColoring(info as AllReduceOpInfo);
    });
  }

  ngOnChanges(changes: SimpleChanges) {
    this.update();
  }

  private createColorInfos() {
    const len = NODE_COLORS.length;
    this.colorInfos = NODE_COLORS.map(
        (color, index) => ({color, label: (index / len).toFixed(1)}));
  }

  private getNodeColor(value: number): string {
    let colorIndex = Math.floor((value || 0) * NODE_COLORS.length);
    colorIndex = Math.min(colorIndex, NODE_COLORS.length - 1);
    return NODE_COLORS[colorIndex];
  }

  private getNodePosition(chipId: number, nodeId: number): ElementInfo {
    const hostWidthWithPadding = HOST_PADDING + BORDER_WIDTH + this.hostWidth +
        BORDER_WIDTH + HOST_PADDING;
    const hostHeightWithPadding = HOST_PADDING + BORDER_WIDTH +
        this.hostHeight + BORDER_WIDTH + HOST_PADDING;
    const chipWidthWithPadding = CHIP_PADDING +
        (BORDER_WIDTH + NODE_WIDTH) * this.nodesPerChip + BORDER_WIDTH +
        CHIP_PADDING;
    const chipHeightWithPadding =
        CHIP_PADDING + BORDER_WIDTH + NODE_HEIGHT + BORDER_WIDTH + CHIP_PADDING;
    let x = CONTAINER_MARGIN + LABEL_PADDING + LABEL_WIDTH + LABEL_PADDING;
    let y = CONTAINER_MARGIN + LABEL_PADDING + LABEL_HEIGHT + LABEL_PADDING;

    x += hostWidthWithPadding *
        Math.floor(
            (chipId % (this.hostColumns * this.hostXStride)) /
            this.hostXStride);
    x += HOST_PADDING + BORDER_WIDTH + HOST_MARGIN;
    x += chipWidthWithPadding * (chipId % this.hostXStride);
    x += CHIP_PADDING + (BORDER_WIDTH + NODE_WIDTH) * nodeId;

    y += hostHeightWithPadding *
        Math.floor(
            Math.floor(chipId / (this.hostColumns * this.hostXStride)) /
            this.hostYStride);
    y += HOST_PADDING + BORDER_WIDTH + HOST_MARGIN;
    y += chipHeightWithPadding *
        (Math.floor(chipId / (this.hostColumns * this.hostXStride)) %
         this.hostYStride);
    y += CHIP_PADDING;

    return {x, y};
  }

  private getNodeRotate(srcX: number, srcY: number, dstX: number, dstY: number):
      number {
    const dx = dstX - srcX;
    const dy = dstY - srcY;
    return Math.atan2(dy, dx);
  }

  private getNodeScale(srcX: number, srcY: number, dstX: number, dstY: number):
      number {
    const dx = srcX - dstX;
    const dy = srcY - dstY;
    return Math.sqrt(dx * dx + dy * dy) / 100;
  }

  private updateArrows() {
    this.arrows = [];

    if (!this.runEnvironment || !this.runEnvironment.topology ||
        !this.podStatsPerCore || !this.channelDb) {
      return;
    }

    const channelInfo = this.channelDb[this.selectedChannelIndex];
    if (!channelInfo || !channelInfo.srcCoreIds || !channelInfo.dstCoreIds) {
      return;
    }

    const len =
        Math.min(channelInfo.srcCoreIds.length, channelInfo.dstCoreIds.length);
    for (let i = 0; i < len; i++) {
      const srcId = channelInfo.srcCoreIds[i] || 0;
      const dstId = channelInfo.dstCoreIds[i] || 0;
      const srcNodeInfo =
          this.getNodePosition(Math.floor(srcId / 2), srcId & 1);
      const dstNodeInfo =
          this.getNodePosition(Math.floor(dstId / 2), dstId & 1);

      this.arrows.push({
        x: dstNodeInfo.x + BORDER_WIDTH + (NODE_WIDTH / 2),
        y: dstNodeInfo.y + BORDER_WIDTH + (NODE_HEIGHT / 2),
        scale: this.getNodeScale(
            srcNodeInfo.x, srcNodeInfo.y, dstNodeInfo.x, dstNodeInfo.y),
        rotate: this.getNodeRotate(
            srcNodeInfo.x, srcNodeInfo.y, dstNodeInfo.x, dstNodeInfo.y),
      });
    }
  }

  private updateChannels() {
    if (!this.runEnvironment || !this.runEnvironment.topology ||
        !this.podStatsPerCore || !this.channelDb) {
      this.channelCount = 0;
      this.firstChannel = 0;
      this.lastChannel = 0;
      this.selectedChannelIndex = 0;
      this.selectedChannelId = 0;
      return;
    }

    this.channels =
        this.channelDb.map(channelInfo => Number(channelInfo.channelId || 0));
    this.channelCount = this.channelDb.length - 1;
    this.firstChannel = this.channels[0];
    this.lastChannel = this.channels[this.channelCount];
    this.selectedChannelIndex = 0;
    this.selectedChannelId = this.channels[0];
    this.updateArrows();
    this.selected.emit(0);
  }

  private updateHosts() {
    this.hosts = [];

    if (!this.runEnvironment || !this.runEnvironment.topology) {
      return;
    }

    const xOffset =
        CONTAINER_MARGIN + LABEL_PADDING + LABEL_WIDTH + LABEL_PADDING;
    const yOffset =
        CONTAINER_MARGIN + LABEL_PADDING + LABEL_HEIGHT + LABEL_PADDING;
    const hostWidthWithPadding = HOST_PADDING + BORDER_WIDTH + this.hostWidth +
        BORDER_WIDTH + HOST_PADDING;
    const hostHeightWithPadding = HOST_PADDING + BORDER_WIDTH +
        this.hostHeight + BORDER_WIDTH + HOST_PADDING;
    for (let i = 0; i < this.hostRows; i++) {
      for (let j = 0; j < this.hostColumns; j++) {
        this.hosts.push({
          x: xOffset + HOST_PADDING + (hostWidthWithPadding * j),
          y: yOffset + HOST_PADDING + (hostHeightWithPadding * i),
        });
      }
    }
  }

  private updateLabels() {
    this.labels = [];

    if (!this.runEnvironment || !this.runEnvironment.topology) {
      return;
    }

    let xOffset =
        CONTAINER_MARGIN + LABEL_PADDING + LABEL_WIDTH + LABEL_PADDING;
    let yOffset = CONTAINER_MARGIN + LABEL_PADDING + LABEL_HEIGHT / 2;
    for (let i = 0; i < this.xDimension; i++) {
      if (i % this.hostXStride === 0) {
        xOffset += HOST_PADDING + BORDER_WIDTH + HOST_MARGIN;
      }
      xOffset += CHIP_PADDING + BORDER_WIDTH + NODE_WIDTH;
      this.labels.push({
        id: i.toString(),
        x: xOffset,
        y: yOffset,
      });
      xOffset += BORDER_WIDTH + NODE_WIDTH + BORDER_WIDTH + CHIP_PADDING;
      if (i % this.hostXStride === this.hostXStride - 1) {
        xOffset += HOST_MARGIN + BORDER_WIDTH + HOST_PADDING;
      }
    }

    xOffset = CONTAINER_MARGIN + LABEL_PADDING + LABEL_WIDTH / 2;
    yOffset = CONTAINER_MARGIN + LABEL_PADDING + LABEL_HEIGHT + LABEL_PADDING;
    for (let i = 0; i < this.yDimension; i++) {
      if (i % this.hostYStride === 0) {
        yOffset += HOST_PADDING + BORDER_WIDTH + HOST_MARGIN;
      }
      yOffset += CHIP_PADDING + BORDER_WIDTH + NODE_HEIGHT / 2;
      this.labels.push({
        id: i.toString(),
        x: xOffset,
        y: yOffset,
      });
      yOffset += NODE_HEIGHT / 2 + BORDER_WIDTH + CHIP_PADDING;
      if (i % this.hostYStride === this.hostYStride - 1) {
        yOffset += HOST_MARGIN + BORDER_WIDTH + HOST_PADDING;
      }
    }
  }

  private updateNodes() {
    this.nodes = [];

    if (!this.runEnvironment || !this.runEnvironment.topology ||
        !this.podStatsPerCore) {
      return;
    }

    Object.keys(this.podStatsPerCore).forEach(coreId => {
      const podStatsRecord = this.podStatsPerCore![coreId];
      const chipId = podStatsRecord.chipId || 0;
      const nodeId = podStatsRecord.nodeId || 0;
      const nodeInfo = this.getNodePosition(chipId, nodeId);
      nodeInfo.id = 'node-' + chipId.toString() + '-' + nodeId.toString();
      if (this.coreIdToReplicaIdMap &&
          this.coreIdToReplicaIdMap[coreId] !== undefined) {
        nodeInfo.rid = this.coreIdToReplicaIdMap[coreId];
      }
      this.nodes.push(nodeInfo);
    });
  }

  private getTpuClass(tpuType: string): TpuClass {
    switch (tpuType) {
      case 'TPU v2':
        return TpuClass.TPU_V2;
      case 'TPU v3':
        return TpuClass.TPU_V3;
      default:
        break;
    }
    return TpuClass.UNKNOWN;
  }

  private updateSystemInfo(tpuType?: string) {
    if (!this.runEnvironment) {
      this.tpuType = '';
      this.xDimension = 0;
      this.yDimension = 0;
      this.containerWidth = 0;
      this.containerHeight = 0;
      this.hostColumns = 0;
      this.hostRows = 0;
      this.hostWidth = 0;
      this.hostHeight = 0;
      return;
    }

    this.tpuType = this.runEnvironment.tpuType || '';

    switch (this.classifyTpuFunc(this.runEnvironment.tpuType || '')) {
      case TpuClass.TPU_V2:
        this.hostXStride = 2;
        this.hostYStride = 2;
        this.nodesPerChip = 2;
        break;
      case TpuClass.TPU_V3:
        this.hostXStride = 4;
        this.hostYStride = 2;
        this.nodesPerChip = 2;
        break;
      default:
        this.hostXStride = 1;
        this.hostYStride = 1;
        this.nodesPerChip = 2;
        console.warn('TPU type: ', tpuType, 'is not supported by pod viewer.');
        break;
    }

    if (!this.runEnvironment.topology) {
      this.xDimension = 0;
      this.yDimension = 0;
      this.containerWidth = 0;
      this.containerHeight = 0;
      this.hostColumns = 0;
      this.hostRows = 0;
      this.hostWidth = 0;
      this.hostHeight = 0;
      return;
    }

    this.xDimension = utils.toNumber(this.runEnvironment.topology.xDimension);
    this.yDimension = utils.toNumber(this.runEnvironment.topology.yDimension);

    const chipWidth = CHIP_PADDING +
        ((BORDER_WIDTH + NODE_WIDTH) * this.nodesPerChip) + BORDER_WIDTH +
        CHIP_PADDING;
    const chipHeight =
        CHIP_PADDING + BORDER_WIDTH + NODE_HEIGHT + BORDER_WIDTH + CHIP_PADDING;
    this.hostWidth = HOST_MARGIN + (chipWidth * this.hostXStride) + HOST_MARGIN;
    this.hostHeight =
        HOST_MARGIN + (chipHeight * this.hostYStride) + HOST_MARGIN;
    const hostWidthWithPadding = HOST_PADDING + BORDER_WIDTH + this.hostWidth +
        HOST_PADDING + BORDER_WIDTH;
    const hostHeightWithPadding = HOST_PADDING + BORDER_WIDTH +
        this.hostHeight + HOST_PADDING + BORDER_WIDTH;

    this.hostColumns = Math.floor(this.xDimension / this.hostXStride);
    if (this.xDimension % this.hostXStride !== 0) {
      this.hostColumns++;
    }
    this.containerWidth = CONTAINER_MARGIN + LABEL_PADDING + LABEL_WIDTH +
        LABEL_PADDING + (hostWidthWithPadding * this.hostColumns) +
        CONTAINER_MARGIN;

    this.hostRows = Math.floor(this.yDimension / this.hostYStride);
    if (this.yDimension % this.hostYStride !== 0) {
      this.hostRows++;
    }
    this.containerHeight = CONTAINER_MARGIN + LABEL_PADDING + LABEL_HEIGHT +
        LABEL_PADDING + (hostHeightWithPadding * this.hostRows) +
        CONTAINER_MARGIN;
  }

  hideTooltip() {
    this.tooltipText = '';
  }

  selectMetric(key: string, label: string) {
    if (!key || !label) {
      this.selectedMetric = '';
      this.selectedMetricLabel = 'Please select a metric';
      return;
    }
    this.selectedMetric = key;
    this.selectedMetricLabel = 'Color: ' + label;

    if (!this.podStatsPerCore) {
      return;
    }

    Object.values(this.podStatsPerCore).forEach(podStatsRecord => {
      const chipId = podStatsRecord.chipId || 0;
      const nodeId = podStatsRecord.nodeId || 0;
      const id = 'node-' + chipId.toString() + '-' + nodeId.toString();
      const nodeEl = this.elRef.nativeElement.querySelector('#' + id);
      if (nodeEl) {
        let value = utils.getPodStatsRecordProperty(podStatsRecord, key);
        value = podStatsRecord.totalDurationUs ?
            value / podStatsRecord.totalDurationUs :
            0;
        nodeEl.style.backgroundColor = this.getNodeColor(value);
      }
    });
  }

  showTooltip(id: string) {
    this.tooltipText = '';
    let podStatsRecord: PodStatsRecord|null = null;
    let coreId = '';

    const found =
        Object.entries(this.podStatsPerCore || {}).find(([key, value]) => {
          const chipId = value.chipId || 0;
          const nodeId = value.nodeId || 0;
          return id === 'node-' + chipId.toString() + '-' + nodeId.toString();
        });

    if (!found || found.length !== 2) {
      return;
    }

    coreId = found[0];
    podStatsRecord = found[1];
    if (!podStatsRecord) {
      return;
    }

    const chipId = podStatsRecord.chipId || 0;
    const nodeId = podStatsRecord.nodeId || 0;
    this.tooltipText += 'pos: (';
    this.tooltipText +=
        (chipId % (this.hostColumns * this.hostXStride)).toString();
    this.tooltipText += ',';
    this.tooltipText +=
        Math.floor(chipId / (this.hostColumns * this.hostXStride)).toString();
    this.tooltipText += ')\n';
    this.tooltipText += 'host: ' + (podStatsRecord.hostName || '') + '\n';
    this.tooltipText += 'chip id: ' + chipId.toString() + '\n';
    this.tooltipText += 'node id: ' + nodeId.toString() + '\n';
    if (this.coreIdToReplicaIdMap &&
        this.coreIdToReplicaIdMap[coreId] !== undefined) {
      this.tooltipText +=
          'replica id: ' + this.coreIdToReplicaIdMap[coreId].toString() + '\n';
    }
    if (this.selectedMetric && this.selectedMetricLabel) {
      const value =
          utils.getPodStatsRecordProperty(podStatsRecord, this.selectedMetric);
      this.tooltipText += this.selectedMetricLabel.replace('Color: ', '');
      this.tooltipText += ' spends ';
      this.tooltipText += value.toFixed(2);
      this.tooltipText += 'us in total, ';
      this.tooltipText += 'taking ';
      this.tooltipText += podStatsRecord.totalDurationUs ?
          (100 * value / podStatsRecord.totalDurationUs).toFixed(2) :
          '0.00';
      this.tooltipText += '% of a step.';
    }

    const nodeInfo = this.getNodePosition(chipId, nodeId);
    this.tooltipX = nodeInfo.x + TOOLTIP_OFFSET_X;
    this.tooltipY = nodeInfo.y + TOOLTIP_OFFSET_Y;
  }

  updateChannelId(id: number) {
    if (id < this.selectedChannelId) {
      let i = this.selectedChannelIndex - 1;
      while (i >= 0) {
        if (this.channels[i] <= id) {
          break;
        }
        i--;
      }
      this.updateChannelIndex(Math.max(0, i));
    } else if (id > this.selectedChannelId) {
      let i = this.selectedChannelIndex + 1;
      while (i <= this.channelCount) {
        if (this.channels[i] >= id) {
          break;
        }
        i++;
      }
      this.updateChannelIndex(Math.min(this.channelCount, i));
    }
  }

  updateChannelIndex(index: number) {
    this.selectedChannelIndex = index;
    this.selectedChannelId = this.channels[index];
    this.updateArrows();
    this.selected.emit(index || 0);
  }

  updateReplicaGroupColoring(info: AllReduceOpInfo) {
    if (!info || !info.replicaGroups || !info.replicaGroups.length) return;
    // Colors the nodes within the same replica group to the same color.
    for (let i = 0; i < info.replicaGroups.length; i++) {
      const group = info.replicaGroups[i].replicaIds;
      if (!group) continue;
      for (let j = 0; j < group.length; j++) {
        const groupEl = this.elRef.nativeElement.querySelectorAll(
            '[rid="' + group[j] + '"]');
        groupEl.forEach((el: HTMLElement) => {
          el.style.backgroundColor = KELLY_COLORS[i % 20];
        });
      }
    }
    this.selectedMetric = '';
  }

  update() {
    this.updateSystemInfo();
    this.updateLabels();
    this.updateHosts();
    this.updateNodes();
    this.selectMetric('', '');
    this.updateChannels();
  }
}
