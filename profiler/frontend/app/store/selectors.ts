import {createFeatureSelector, createSelector} from '@ngrx/store';

import {AppState, MemoryViewerState, OpProfileState, PodViewerState, STORE_KEY} from './state';

const appState = createFeatureSelector<AppState>(STORE_KEY);

/** Selector for MemoryViewerState */
export const getMemoryViewerState: any = createSelector(
    appState, (appState: AppState) => appState.memoryViewerState);

/** Selector for ActiveHeapObjectState */
export const getActiveHeapObjectState: any = createSelector(
    getMemoryViewerState,
    (memoryViewerState: MemoryViewerState) =>
        memoryViewerState.activeHeapObject);

/** Selector for OpProfileState */
export const getOpProfileState: any =
    createSelector(appState, (appState: AppState) => appState.opProfileState);

/** Selector for ActiveOpProfileNodeState */
export const getActiveOpProfileNodeState: any = createSelector(
    getOpProfileState,
    (opProfileState: OpProfileState) => opProfileState.activeOpProfileNode);

/** Selector for PodViewerState */
export const getPodViewerState: any =
    createSelector(appState, (appState: AppState) => appState.podViewerState);

/** Selector for ActivePodViewerInfoState */
export const getActivePodViewerInfoState: any = createSelector(
    getPodViewerState,
    (podViewerState: PodViewerState) => podViewerState.activePodViewerInfo);

/** Selector for CapturingProfileState */
export const getCapturingProfileState: any =
    createSelector(appState, (appState: AppState) => appState.capturingProfile);

/** Selector for LoadingState */
export const getLoadingState: any =
    createSelector(appState, (appState: AppState) => appState.loadingState);

/** Selector for CurrentTool */
export const getCurrentTool: any =
    createSelector(appState, (appState: AppState) => appState.currentTool);
