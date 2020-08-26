import {Action, createReducer, on} from '@ngrx/store';

import * as actions from './actions';
import {AppState, INIT_APP_STATE} from './state';

export const reducer: any = createReducer(
    INIT_APP_STATE,
    on(
        actions.setActiveHeapObjectAction,
        (state, action) => {
          return {
            ...state,
            memoryViewerState: {
              ...state.memoryViewerState,
              activeHeapObject: (action as any).activeHeapObject,
            }
          };
        },
        ),
    on(
        actions.setActiveOpProfileNodeAction,
        (state: AppState, action) => {
          return {
            ...state,
            opProfileState: {
              ...state.opProfileState,
              activeOpProfileNode: (action as any).activeOpProfileNode,
            }
          };
        },
        ),
    on(
        actions.setActivePodViewerInfoAction,
        (state: AppState, action) => {
          return {
            ...state,
            podViewerState: {
              ...state.podViewerState,
              activePodViewerInfo: (action as any).activePodViewerInfo,
            }
          };
        },
        ),
    on(
        actions.setCapturingProfileAction,
        (state: AppState, action) => {
          return {
            ...state,
            capturingProfile: (action as any).capturingProfile,
          };
        },
        ),
    on(
        actions.setLoadingStateAction,
        (state: AppState, action) => {
          return {
            ...state,
            loadingState: (action as any).loadingState,
          };
        },
        ),
    on(
        actions.setCurrentToolStateAction,
        (state: AppState, action) => {
          return {
            ...state,
            currentTool: (action as any).currentTool,
          };
        },
        ),
);

/** Reducer */
export function rootReducer(state: AppState|undefined, action: Action) {
  return reducer(state, action);
}
