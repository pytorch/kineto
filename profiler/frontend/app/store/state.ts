import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';

/** Type for active heap object state */
type ActiveHeapObjectState = HeapObject|null;

/** State of memory viewer */
export interface MemoryViewerState {
  activeHeapObject: ActiveHeapObjectState;
}

/** Type for active op profile node state */
type ActiveOpProfileNodeState = Node|null;

/** State of op profile */
export interface OpProfileState {
  activeOpProfileNode: ActiveOpProfileNodeState;
}

/** Type for active pod viewer info state */
type ActivePodViewerInfoState = AllReduceOpInfo|ChannelInfo|PodStatsRecord|null;

/** State of pod viewer */
export interface PodViewerState {
  activePodViewerInfo: ActivePodViewerInfoState;
}

/** Type for capturing profile state */
type CapturingProfileState = boolean;

/** State of loading */
export interface LoadingState {
  loading: boolean;
  message: string;
}

/** Type for current tool state */
type CurrentToolState = string;

/** State object */
export interface AppState {
  memoryViewerState: MemoryViewerState;
  opProfileState: OpProfileState;
  podViewerState: PodViewerState;
  capturingProfile: CapturingProfileState;
  loadingState: LoadingState;
  currentTool: CurrentToolState;
}

/** Initial state of active heap object */
const INIT_ACTIVE_HEAP_OBJECT_STATE: ActiveHeapObjectState = null;

/** Initial state object */
export const INIT_MEMORY_VIEWER_STATE: MemoryViewerState = {
  activeHeapObject: INIT_ACTIVE_HEAP_OBJECT_STATE,
};

/** Initial state of active op profile node */
const INIT_ACTIVE_OP_PROFILE_NODE_STATE: ActiveOpProfileNodeState = null;

/** Initial state of op profile */
export const INIT_OP_PROFILE_STATE: OpProfileState = {
  activeOpProfileNode: INIT_ACTIVE_OP_PROFILE_NODE_STATE,
};

/** Initial state of active pod viewer info */
const INIT_ACTIVE_POD_VIEWER_INFO_STATE: ActivePodViewerInfoState = null;

/** Initial state of pod viewer */
export const INIT_POD_VIEWER_STATE: PodViewerState = {
  activePodViewerInfo: INIT_ACTIVE_POD_VIEWER_INFO_STATE,
};

/** Initial state of capturing profile */
const INIT_CAPTURING_PROFILE_STATE: CapturingProfileState = false;

/** Initial state of loading */
export const INIT_LOADING_STATE: LoadingState = {
  loading: false,
  message: '',
};

/** Initial state of current tool */
const INIT_CURRENT_TOOL_STATE: CurrentToolState = '';

/** Initial state object */
export const INIT_APP_STATE: AppState = {
  memoryViewerState: INIT_MEMORY_VIEWER_STATE,
  opProfileState: INIT_OP_PROFILE_STATE,
  podViewerState: INIT_POD_VIEWER_STATE,
  capturingProfile: INIT_CAPTURING_PROFILE_STATE,
  loadingState: INIT_LOADING_STATE,
  currentTool: INIT_CURRENT_TOOL_STATE,
};

/** Feature key for store */
export const STORE_KEY = 'root';
