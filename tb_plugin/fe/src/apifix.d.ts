/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { OperatorGraph } from './api/generated/api'

declare module './api/generated/api' {
  export interface OperatorGraph {
    device_total_time: OperatorGraph['deviceTotalTime']
    device_self_time: OperatorGraph['deviceSelfTime']
    host_total_time: OperatorGraph['hostTotalTime']
    host_self_time: OperatorGraph['hostSelfTime']
  }
}
