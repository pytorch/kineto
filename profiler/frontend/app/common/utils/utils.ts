import {PodStatsRecord, SimpleDataTableOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {OpProfileNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_node';
import {Store} from '@ngrx/store';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';

const PRIMITIVE_TYPE_BYTE_SIZE: {[key: string]: number} = {
  'BF16': 2,
  'C64': 8,
  'F16': 2,
  'F32': 4,
  'F64': 8,
  'PRED': 1,
  'TOKEN': 0,
  'S8': 1,
  'S16': 2,
  'S32': 4,
  'S64': 8,
  'U8': 1,
  'U16': 2,
  'U32': 4,
  'U64': 8,
};

const SHUFFLED_MATERIAL_COLORS = [
  '#e91e63', '#2196f3', '#81c784', '#4dd0e1', '#3f51b5', '#e53935', '#ff9100',
  '#b39ddb', '#90a4ae', '#26c6da', '#ad1457', '#03a9f4', '#2196f3', '#c2185b',
  '#795548', '#f9a825', '#00bfa5', '#880e4f', '#d500f9', '#ce93d8', '#ec407a',
  '#4caf50', '#ff8f00', '#ffca28', '#ab47bc', '#00e5ff', '#ff9800', '#40c4ff',
  '#1e88e5', '#9fa8da', '#bf360c', '#00b8d4', '#f57f17', '#64b5f6', '#e040fb',
  '#ffab91', '#4caf50', '#01579b', '#66bb6a', '#ef9a9a', '#558b2f', '#fb8c00',
  '#ff4081', '#00e676', '#388e3c', '#424242', '#6d4c41', '#c62828', '#616161',
  '#00897b', '#448aff', '#0d47a1', '#607d8b', '#673ab7', '#00c853', '#2e7d32',
  '#ffa726', '#5e35b1', '#ba68c8', '#8d6e63', '#00bcd4', '#ff6f00', '#f4511e',
  '#ff1744', '#9e9e9e', '#d81b60', '#4a148c', '#26a69a', '#689f38', '#7b1fa2',
  '#b0bec5', '#304ffe', '#f48fb1', '#ffd600', '#ffb74d', '#8bc34a', '#303f9f',
  '#5d4037', '#80cbc4', '#ffcc80', '#00acc1', '#3e2723', '#ff5252', '#ff7043',
  '#e91e63', '#ea80fc', '#e65100', '#d84315', '#212121', '#ff5722', '#1976d2',
  '#2962ff', '#bdbdbd', '#3949ab', '#69f0ae', '#d50000', '#ffd740', '#c0ca33',
  '#ff6e40', '#00b0ff', '#2979ff', '#e64a19', '#7c4dff', '#607d8b', '#009688',
  '#ffb300', '#c51162', '#ffc400', '#29b6f6', '#3d5afe', '#76ff03', '#cddc39',
  '#b388ff', '#5c6bc0', '#9e9d24', '#7cb342', '#ef5350', '#fdd835', '#ef6c00',
  '#4fc3f7', '#6200ea', '#004d40', '#ff8a65', '#ffab00', '#80deea', '#0097a7',
  '#7e57c2', '#ff6d00', '#1565c0', '#455a64', '#ffc107', '#4527a0', '#ff5722',
  '#f44336', '#f57c00', '#827717', '#a5d6a7', '#82b1ff', '#9c27b0', '#ff80ab',
  '#e1bee7', '#78909c', '#311b92', '#00695c', '#4e342e', '#3f51b5', '#651fff',
  '#9e9e9e', '#81d4fa', '#f8bbd0', '#b71c1c', '#0091ea', '#673ab7', '#a1887f',
  '#4db6ac', '#ffa000', '#6a1b9a', '#43a047', '#bcaaa4', '#546e7a', '#aeea00',
  '#e57373', '#ffccbc', '#006064', '#fbc02d', '#ffeb3b', '#8bc34a', '#039be5',
  '#8e24aa', '#80d8ff', '#009688', '#9ccc65', '#512da8', '#ffc107', '#757575',
  '#0277bd', '#ff3d00', '#33691e', '#03a9f4', '#00838f', '#ff8a80', '#283593',
  '#f50057', '#1a237e', '#90caf9', '#9c27b0', '#aa00ff', '#aed581', '#afb42b',
  '#9575cd', '#d32f2f', '#64dd17', '#f44336', '#795548', '#cddc39', '#ff9e80',
  '#7986cb', '#dd2c00', '#0288d1', '#ff9800', '#263238', '#00796b', '#42a5f5',
  '#8c9eff', '#1b5e20', '#ffab40', '#536dfe', '#00bcd4', '#f06292',
];

const KNOWN_TOOLS = [
  'input_pipeline_analyzer',
  'memory_profile',
  'memory_viewer',
  'op_profile',
  'pod_viewer',
  'tensorflow_stats',
  'trace_viewer',
];

/**
 * Returns the number of bytes of the primitive type.
 */
export function byteSizeOfPrimitiveType(type: string): number {
  if (!PRIMITIVE_TYPE_BYTE_SIZE.hasOwnProperty(type)) {
    console.error('Unhandled primitive type ' + type);
    return 0;
  }
  return PRIMITIVE_TYPE_BYTE_SIZE[type];
}

/**
 * Converts from number of bytes to MiB.
 */
export function bytesToMiB(numBytes: number): number {
  return numBytes / (1024 * 1024);
}

/**
 * Returns a color for chart item by index.
 */
export function getChartItemColorByIndex(index: number): string {
  return SHUFFLED_MATERIAL_COLORS[index % SHUFFLED_MATERIAL_COLORS.length];
}

/**
 * Converts from string to number.
 */
export function toNumber(value: string|undefined): number {
  const num = Number(value);
  return isNaN(num) ? 0 : num;
}

/**
 * Returns a rgba string.
 */
function rgba(red: number, green: number, blue: number, alpha: number): string {
  return 'rgba(' + Math.round(red * 255).toString() + ',' +
      Math.round(green * 255).toString() + ',' +
      Math.round(blue * 255).toString() + ',' + alpha.toString() + ')';
}

/**
 * Computes a flame color.
 */
export function flameColor(
    fraction: number, brightness?: number, opacity?: number,
    curve?: Function): string {
  if (brightness === void 0) {
    brightness = 1;
  }
  if (opacity === void 0) {
    opacity = 1;
  }
  if (curve === void 0) {
    curve = (x: number) => 1 - Math.sqrt(1 - x);
  }
  if (isNaN(fraction)) {
    return rgba(brightness, brightness, brightness, opacity);
  }
  fraction = curve(fraction);

  // Or everythin is depressing and red.
  return fraction < 0.5 ?
      rgba(brightness, 2 * fraction * brightness, 0, opacity) :
      rgba(2 * (1 - fraction) * brightness, brightness, 0, opacity);
}

/**
 * Computes a Flops color.
 */
export function flopsColor(fraction: number): string {
  return flameColor(Math.min(fraction, 1), 0.7, 1, Math.sqrt);
}

/**
 * Computes a memory bandwidth color.
 */
export function bwColor(fraction: number): string {
  return flameColor(Math.max(1 - fraction, 0), 0.7, 1, Math.sqrt);
}

/**
 * Computes an utilization for fused operations.
 */
export function utilization(node: OpProfileNode): number {
  // NaN indicates undefined utilization for fused operations (we can't
  // measure performance inside a fusion). It could also indicate operations
  // with zero time, but they currently don't appear in the profile.
  if (!node || !node.metrics || !node.metrics.time) return NaN;
  return (node.metrics.flops || 0) / node.metrics.time;
}

/**
 * Computes a memory utilization.
 */
export function memoryUtilization(node: OpProfileNode): number {
  // NaN indicates undefined memory utilization (th profile was collected
  // from older versions of profiler).
  if (!node || !node.metrics || !node.metrics.memoryBandwidth) return NaN;
  return node.metrics.memoryBandwidth;
}

/**
 * Returns whether a node has flops.
 */
export function hasFlops(node: OpProfileNode): boolean {
  return !!node && !!node.metrics && !!node.metrics.time;
}

/**
 * Returns whether a node has a memory utilization.
 */
export function hasMemoryUtilization(node: OpProfileNode): boolean {
  return !!node && !!node.metrics && !!node.metrics.memoryBandwidth;
}

/**
 * Computes a percent.
 */
export function percent(fraction: number): string {
  if (isNaN(fraction)) return '-';
  return fraction >= 1.995 ?
      '200%' :
      fraction < 0.00001 ?
      '0.00%' :
      Number((fraction * 100).toPrecision(2)).toString() + '%';
}

/**
 * Computes wasted time.
 */
export function timeWasted(node: OpProfileNode): number {
  if (!node || !node.metrics) return NaN;
  return (
      (node.metrics.time || 0) *
      (1 - Math.max(utilization(node), memoryUtilization(node))));
}

/**
 * Returns podStatsRecord property.
 */
export function getPodStatsRecordProperty(
    podStatsRecord: PodStatsRecord, key: string): number {
  return (podStatsRecord as {[key: string]: number})[key] || 0;
}

/**
 * Scrolls to the bottom of sidenav to show detail views.
 */
export function scrollBottomOfSidenav() {
  const sidenavContainer =
      document.querySelector('.mat-drawer-inner-container');
  if (sidenavContainer) {
    setTimeout(() => {
      sidenavContainer.scrollTo(0, sidenavContainer.scrollHeight);
    }, 1);
  }
}

/**
 * Returns a string with an anchor tag.
 */
export function addAnchorTag(value: string = ''): string {
  return '<a>' + value + '</a>';
}

/**
 * Returns a string with the known tool name changed to an anchor tag.
 */
export function convertKnownToolToAnchorTag(value: string = ''): string {
  KNOWN_TOOLS.forEach(tool => {
    value = value.replace(new RegExp(tool, 'g'), addAnchorTag(tool));
  });
  return value;
}

/**
 * Parse diagnostics data table and returns Diagnostics object.
 */
export function parseDiagnosticsDataTable(
    diagnosticsTable: SimpleDataTableOrNull): Diagnostics {
  const diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  if (!diagnosticsTable || !diagnosticsTable.rows) return diagnostics;
  /** Convert data table to string arrays */
  diagnosticsTable.rows.forEach(row => {
    if (String(row.c![0].v!) === 'ERROR') {
      diagnostics.errors.push(String(row.c![1].v!));
    } else if (String(row.c![0].v!) === 'WARNING') {
      diagnostics.warnings.push(String(row.c![1].v!));
    } else {
      diagnostics.info.push(String(row.c![1].v!));
    }
  });
  return diagnostics;
}

/**
 * Returns the column index found by the given label.
 */
export function getColumnIndex(
    dataTable: SimpleDataTableOrNull, label: string): number {
  if (!dataTable || !dataTable.cols) {
    return -1;
  }

  for (let i = 0; i < dataTable.cols.length; i++) {
    if (dataTable.cols[i].label === label) {
      return i;
    }
  }

  return -1;
}

/**
 * Sets the global loading state.
 */
export function setLoadingState(
    beingLoaded: boolean, store: Store<{}>) {
  if (beingLoaded) {
    store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading data',
      }
    }));
  } else {
    store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: false,
        message: '',
      }
    }));
  }
}
