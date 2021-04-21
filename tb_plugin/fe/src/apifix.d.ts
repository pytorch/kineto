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
  }

  export interface CallStackTableDataInner {
    input_shape: CallStackTableDataInner['inputShape']
    device_self_duration: CallStackTableDataInner['deviceSelfDuration']
    device_total_duration: CallStackTableDataInner['deviceTotalDuration']
    host_self_duration: CallStackTableDataInner['hostSelfDuration']
    host_total_duration: CallStackTableDataInner['hostTotalDuration']
    call_stack: CallStackTableDataInner['callStack']
  }
}
