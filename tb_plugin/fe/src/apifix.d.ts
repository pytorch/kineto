/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { OperatorGraph, OperationTableDataInner } from './api/generated/api'

declare module './api/generated/api' {
  export interface OperatorGraph {
    device_total_time: OperatorGraph['deviceTotalTime']
    device_self_time: OperatorGraph['deviceSelfTime']
    host_total_time: OperatorGraph['hostTotalTime']
    host_self_time: OperatorGraph['hostSelfTime']
  }

  export interface OperationTableDataInner {
    input_shape: OperationTableDataInner['inputShape']
    device_self_duration: OperationTableDataInner['deviceSelfDuration']
    device_total_duration: OperationTableDataInner['deviceTotalDuration']
    host_self_duration: OperationTableDataInner['hostSelfDuration']
    host_total_duration: OperationTableDataInner['hostTotalDuration']
    has_call_stack: OperationTableDataInner['hasCallStack']
    tc_eligible: OperationTableDataInner['tcEligible']
    tc_self_ratio: OperationTableDataInner['tcSelfRatio']
    tc_total_ratio: OperationTableDataInner['tcTotalRatio']
  }

  export interface CallStackTableDataInner {
    input_shape: CallStackTableDataInner['inputShape']
    device_self_duration: CallStackTableDataInner['deviceSelfDuration']
    device_total_duration: CallStackTableDataInner['deviceTotalDuration']
    host_self_duration: CallStackTableDataInner['hostSelfDuration']
    host_total_duration: CallStackTableDataInner['hostTotalDuration']
    call_stack: CallStackTableDataInner['callStack']
    tc_eligible: OperationTableDataInner['tcEligible']
    tc_self_ratio: OperationTableDataInner['tcSelfRatio']
    tc_total_ratio: OperationTableDataInner['tcTotalRatio']
  }

  export interface Overview {
    gpu_metrics: Overview['gpuMetrics']
  }

  export interface MemoryStatsTableMetadata {
    default_device: MemoryStatsTableMetadata['defaultDevice']
  }

  export interface MemoryCurveDataMetadata {
    default_device: MemoryCurveDataMetadata['defaultDevice']
    first_ts: MemoryCurveDataMetadata['firstTs']
    time_metric: MemoryCurveDataMetadata['timeMetric']
    memory_metric: MemoryCurveDataMetadata['memoryMetric']
    time_factor: MemoryCurveDataMetadata['timeFactor']
    memory_factor: MemoryCurveDataMetadata['memoryFactor']
  }
}
