import {createAction, props} from '@ngrx/store';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';

import {LoadingState} from './state';

/** Action to set active heap object */
export const setActiveHeapObjectAction: any = createAction(
    '[Heap object] Set active heap object',
    props<{activeHeapObject: HeapObject | null}>(),
);

/** Action to set active op profile node */
export const setActiveOpProfileNodeAction: any = createAction(
    '[Op Profile Node] Set active op profile node',
    props<{activeOpProfileNode: Node | null}>(),
);

/** Action to set active info of the pod viewer */
export const setActivePodViewerInfoAction: any = createAction(
    '[Pod Viewer Info] Set active pod viewer info',
    props<{
      activePodViewerInfo: AllReduceOpInfo | ChannelInfo | PodStatsRecord | null
    }>(),
);

/** Action to set capturing profile */
export const setCapturingProfileAction: any = createAction(
    '[App State] Set capturing profile',
    props<{capturingProfile: boolean}>(),
);

/** Action to set loading state */
export const setLoadingStateAction: any = createAction(
    '[App State] Set loading state',
    props<{loadingState: LoadingState}>(),
);

/** Action to set current tool state */
export const setCurrentToolStateAction: any = createAction(
    '[App State] Set current tool state',
    props<{currentTool: string}>(),
);
