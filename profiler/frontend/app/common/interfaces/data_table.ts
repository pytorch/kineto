import * as hloProto from 'org_xprof/frontend/app/common/interfaces/hlo.proto';
import * as diagnosticsProto from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import * as memoryProfileProto from 'org_xprof/frontend/app/common/interfaces/memory_profile.proto';
import * as opProfileProto from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';

/** The base interface for a cell.  */
declare interface Cell<T> {
  v?: T;
}

/** All cell type. */
export type DataTableCell = Cell<boolean|number|string>;

/** The base interface for a colume role. */
declare interface ColumeRole {
  role?: string;
}

/** The base interface for a column. */
declare interface DataTableColumn {
  id?: string;
  label?: string;
  type?: string;
  p?: ColumeRole;
}

/** The base interface for a row value. */
declare interface DataTableRow {
  c?: DataTableCell[];
}

/** The base interface for an empty property. */
declare interface EmptyProperty {}

/** The base interface for data table without perperty. */
export declare interface SimpleDataTable {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: EmptyProperty;
}

/** The data table type for data table without perperty or null. */
export type SimpleDataTableOrNull = SimpleDataTable|null;

/* tslint:disable enforce-name-casing */
/** The base interface for a property of general analysis. */
declare interface GeneralAnalysisProperty {
  device_idle_time_percent?: string;
  device_duty_cycle_percent?: string;
  flop_rate_utilization_relative_to_roofline?: string;
  host_idle_time_percent?: string;
  memory_bw_utilization_relative_to_hw_limit?: string;
  mxu_utilization_percent?: string;
  host_tf_op_percent?: string;
  device_tf_op_percent?: string;
  host_op_time_eager_percent?: string;
  device_op_time_eager_percent?: string;
  device_compute_16bit_percent?: string;
  device_compute_32bit_percent?: string;
  remark_color?: string;
  remark_text?: string;
}
/* tslint:enable */

/* tslint:disable enforce-name-casing */
/** The base interface for properties of meta host-op table. */
declare interface MetaHostOpTableProperty {
  num_host_op_tables: string;
  valid_host_ops: string;
  hostnames: string;
  values: string;
}
/* tslint:enable */

/** The base interface for meta host-op table. */
export declare interface MetaHostOpTable {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p: MetaHostOpTableProperty;
}

/** MetaHostOpTable type or Null. */
export type MetaHostOpTableOrNull = MetaHostOpTable|null;

/** The base interface for properties of host-op table. */
declare interface HostOpTableProperty {
  hostop: string;
  hostname: string;
  value: string;
}

/** The base interface for host-op table. */
export declare interface HostOpTable {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p: HostOpTableProperty;
}

/** HostOpTable type or Null. */
export type HostOpTableOrNull = HostOpTable|null;

/** The base interface for a general analysis. */
export declare interface GeneralAnalysis {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: GeneralAnalysisProperty;
}

/** The data table type for a general analysis or null. */
export type GeneralAnalysisOrNull = GeneralAnalysis|null;

/* tslint:disable enforce-name-casing */
/** The base interface for a property of input pipeline analysis. */
declare interface InputPipelineAnalysisProperty {
  hardware_type?: string;
  steptime_ms_average?: string;
  steptime_ms_standard_deviation?: string;
  steptime_ms_minimum?: string;
  steptime_ms_maximum?: string;
  infeed_percent_average?: string;
  infeed_percent_standard_deviation?: string;
  infeed_percent_minimum?: string;
  infeed_percent_maximum?: string;
  idle_ms_average?: string;
  input_ms_average?: string;
  compute_ms_average?: string;
  input_conclusion?: string;
  output_conclusion?: string;
  summary_nextstep?: string;
  other_time_ms_avg?: string;
  other_time_ms_sdv?: string;
  compile_time_ms_avg?: string;
  compile_time_ms_sdv?: string;
  outfeed_time_ms_avg?: string;
  outfeed_time_ms_sdv?: string;
  infeed_time_ms_avg?: string;
  infeed_time_ms_sdv?: string;
  kernel_launch_time_ms_avg?: string;
  kernel_launch_time_ms_sdv?: string;
  host_compute_time_ms_avg?: string;
  host_compute_time_ms_sdv?: string;
  device_to_device_time_ms_avg?: string;
  device_to_device_time_ms_sdv?: string;
  device_compute_time_ms_avg?: string;
  device_compute_time_ms_sdv?: string;
}
/* tslint:enable */

/** The base interface for an input pipeline analysis. */
export declare interface InputPipelineAnalysis {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: InputPipelineAnalysisProperty;
}

/** The data table type for an input pipeline analysis or null. */
export type InputPipelineAnalysisOrNull = InputPipelineAnalysis|null;

/** The base interface for a top ops table column. */
export declare interface TopOpsColumn {
  selfTimePercent: number;
  cumulativeTimePercent: number;
  category: number;
  operation: number;
  flopRate: number;
  tcEligibility: number;
  tcUtilization: number;
}

/* tslint:disable enforce-name-casing */
/** The base interface for a property of run environment. */
declare interface RunEnvironmentProperty {
  error_message?: string;
  host_count?: string;
  device_core_count?: string;
  device_type?: string;
}
/* tslint:enable */

/** The base interface for a run environment. */
export declare interface RunEnvironment {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: RunEnvironmentProperty;
}

/** The data table type for a run environment or null. */
export type RunEnvironmentOrNull = RunEnvironment|null;

/* tslint:disable enforce-name-casing */
/** The base interface for a property of recommendation result. */
declare interface RecommendationResultProperty {
  bottleneck?: string;
  statement?: string;
  tf_function_statement_html?: string;
  eager_statement_html?: string;
  all_other_bottleneck?: string;
  all_other_statement?: string;
  kernel_launch_bottleneck?: string;
  kernel_launch_statement?: string;
  precision_statement?: string;
}
/* tslint:enable */

/** The base interface for a recommendation result. */
export declare interface RecommendationResult {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: RecommendationResultProperty;
}

/** The data table type for a recommendation result or null. */
export type RecommendationResultOrNull = RecommendationResult|null;

/* tslint:disable enforce-name-casing */
/** The base interface for a property of normalized accelerator performance. */
declare interface NormalizedAcceleratorPerformanceProperty {
  background_link_0?: string;
  background_link_1?: string;
  inference_cost_line_0?: string;
  inference_cost_line_1?: string;
  inference_productivity_line_0?: string;
  inference_productivity_line_1?: string;
  total_naps_line_0?: string;
  total_naps_line_1?: string;
  total_naps_line_2?: string;
  training_cost_line_0?: string;
  training_cost_line_1?: string;
  training_productivity_line_0?: string;
  training_productivity_line_1?: string;
}
/* tslint:enable */

/** The base interface for a normalized accelerator performance. */
export declare interface NormalizedAcceleratorPerformance {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: NormalizedAcceleratorPerformanceProperty;
}

/** The data table type for a normalized accelerator performance or null. */
export type NormalizedAcceleratorPerformanceOrNull =
    NormalizedAcceleratorPerformance|null;

/** The data table type for an input pipeline device-side analysis. */
export type InputPipelineDeviceAnalysis = InputPipelineAnalysis;

/** The data table type for an input pipeline device-side analysis or null. */
export type InputPipelineDeviceAnalysisOrNull =
    InputPipelineDeviceAnalysis|null;

/* tslint:disable enforce-name-casing */
/** The base interface for a property of input pipeline host-side anaysis. */
declare interface InputPipelineHostAnalysisProperty {
  advanced_file_read_us?: string;
  demanded_file_read_us?: string;
  enqueue_us?: string;
  preprocessing_us?: string;
  unclassified_nonequeue_us?: string;
}
/* tslint:enable */

/** The base interface for an input pipeline host-side analysis. */
export declare interface InputPipelineHostAnalysis {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: InputPipelineHostAnalysisProperty;
}

/** The data table type for an input pipeline host-side analysis or null. */
export type InputPipelineHostAnalysisOrNull = InputPipelineHostAnalysis|null;

/** The base interface for a host ops table column. */
export interface HostOpsColumn {
  opName: number;
  count: number;
  timeInMs: number;
  timeInPercent: number;
  selfTimeInMs: number;
  selfTimeInPercent: number;
  category: number;
}

/* tslint:disable enforce-name-casing */
/** The base interface for a property of tensorflow stats. */
declare interface TensorflowStatsProperty {
  architecture_type?: string;
  device_tf_pprof_link?: string;
  host_tf_pprof_link?: string;
  task_type?: string;
}
/* tslint:enable */

/** The base interface for a tensorflow stats. */
export declare interface TensorflowStatsData {
  cols?: DataTableColumn[];
  rows?: DataTableRow[];
  p?: TensorflowStatsProperty;
}

/** The base interface for a replica group. */
declare interface ReplicaGroup {
  replicaIds?: /* int64 */ string[];
}

/** The base interface for all reduce op info . */
export declare interface AllReduceOpInfo {
  name?: string;
  occurrences?: /* uint32 */ number;
  durationUs?: /* double */ number;
  dataSize?: /* uint64 */ string;
  replicaGroups?: ReplicaGroup[];
  description?: string;
}

/** The base interface for a channel info. */
export declare interface ChannelInfo {
  channelId?: /* int64 */ string;
  srcCoreIds?: /* uint32 */ number[];
  dstCoreIds?: /* uint32 */ number[];
  dataSize?: /* uint64 */ string;
  durationUs?: /* double */ number;
  occurrences?: /* uint32 */ number;
  utilization?: /* double */ number;
  hloNames?: string[];
  sendDelayUs?: /* double */ number;
  description?: string;
}

/** The base interface for a pod stats record. */
export declare interface PodStatsRecord {
  hostName?: string;
  chipId?: /* int32 */ number;
  nodeId?: /* int32 */ number;
  stepNum?: /* int32 */ number;
  totalDurationUs?: /* double */ number;
  highFlopsComputeUs?: /* double */ number;
  lowFlopsComputeUs?: /* double */ number;
  hostInfeedDurationUs?: /* double */ number;
  hostOutfeedDurationUs?: /* double */ number;
  sendDurationUs?: /* double */ number;
  recvDurationUs?: /* double */ number;
  allReduceComputeDurationUs?: /* double */ number;
  allReduceSyncDurationUs?: /* double */ number;
  bottleneck?: string;
}

/** The base interface for a hlo info. */
declare interface HloInfo {
  category?: string;
  description?: string;
}

/** The base interface for a pod stats map. */
declare interface PodStatsMap {
  stepNum?: /* uint32 */ number;
  podStatsPerCore?: {[key: /* uint32 */ string]: PodStatsRecord};
  channelDb?: ChannelInfo[];
  coreIdToReplicaIdMap?: {[key: /* uint32 */ string]: /* uint32 */ number};
  allReduceOpDb?: AllReduceOpInfo[];
}

/** The base interface for a pod stats sequence. */
declare interface PodStatsSequence {
  podStatsMap?: PodStatsMap[];
}

/** The base interface for a system topology. */
declare interface SystemTopology {
  xDimension?: /* int64 */ string;
  yDimension?: /* int64 */ string;
  zDimension?: /* int64 */ string;
  numExpectedReducedChips?: /* int64 */ string;
}

/** The base interface for a pod viewer run environment. */
export declare interface PodViewerRunEnvironment {
  tpuType?: string;
  topology?: SystemTopology;
}

/** The base interface for a pod viewer summary. */
export declare interface PodViewerSummary {
  warnings?: string[];
}

/** The base interface for a pod viewer database. */
export declare interface PodViewerDatabase {
  podStatsSequence?: PodStatsSequence;
  runEnvironment?: PodViewerRunEnvironment;
  hloInfoMap?: {[key: string]: HloInfo};
  summary?: PodViewerSummary;
  diagnostics?: diagnosticsProto.Diagnostics;
}

/** The data table type for a tensorflow stats or null. */
export type TensorflowStatsDataOrNull = TensorflowStatsData|null;

/** The data table type for a HloProto or null. */
export type HloProtoOrNull = (hloProto.HloProto)|null;

/** The data table type for a MemoryProfile or null. */
export type MemoryProfileProtoOrNull = (memoryProfileProto.MemoryProfile)|null;

/** The data table type for a MemoryProfileSnapshot. */
export type MemoryProfileSnapshot = memoryProfileProto.MemoryProfileSnapshot;

/** The data table type for a Profile or null. */
export type ProfileOrNull = (opProfileProto.Profile)|null;

/** All overview page data table type. */
export type OverviewDataTable =
    GeneralAnalysis|InputPipelineAnalysis|RecommendationResult|RunEnvironment|
    SimpleDataTable|NormalizedAcceleratorPerformance;

/** All overview page data tuple type. */
export type OverviewDataTuple = [
  GeneralAnalysisOrNull,
  InputPipelineAnalysisOrNull,
  RunEnvironmentOrNull,
  RecommendationResultOrNull,
  SimpleDataTableOrNull,
  NormalizedAcceleratorPerformanceOrNull,
  SimpleDataTableOrNull,
];

/** All input pipeline page data table type. */
export type InputPipelineDataTable =
    InputPipelineDeviceAnalysis|InputPipelineHostAnalysis|MetaHostOpTable|HostOpTable|SimpleDataTable;

/** The data table type for a PodViewerDatabase or null. */
export type PodViewerDatabaseOrNull = PodViewerDatabase|null;

/** The data types with number, string, or undefined. */
export type PrimitiveTypeNumberStringOrUndefined = number|string|undefined;

/** All data table type. */
export type DataTable =
    OverviewDataTable[]|InputPipelineDataTable[]|TensorflowStatsData[]|
    hloProto.HloProto|memoryProfileProto.MemoryProfile|opProfileProto.Profile|
    PodViewerDatabase|null;
