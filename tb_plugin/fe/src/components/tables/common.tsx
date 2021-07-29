/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { firstOrUndefined, isDef } from '../../utils/def'
import { CallStackTableDataInner, OperationTableDataInner } from '../../api'
import type { ColumnsType } from 'antd/es/table'
import { ClassNameMap } from '@material-ui/styles'

export function getCommonOperationColumns<
  T extends OperationTableDataInner | CallStackTableDataInner
>(
  data: T[] | undefined,
  defaultSort?: string,
  tooltips?: any,
  classes?: ClassNameMap<'tooltip'>
): ColumnsType<T> {
  const firstData = firstOrUndefined(data)

  const hasInputShape = !firstData || isDef(firstData.input_shape)
  const hasDeviceSelfDuration =
    !firstData || isDef(firstData.device_self_duration)
  const hasDeviceTotalDuration =
    !firstData || isDef(firstData.device_total_duration)
  const hasTcEligible = !firstData || isDef(firstData.tc_eligible)
  const hasTcSelfRatio = !firstData || isDef(firstData.tc_self_ratio)
  const hasTcTotalRatio = !firstData || isDef(firstData.tc_total_ratio)

  const nameCompare = (a: T, b: T) => a.name.localeCompare(b.name)
  const callsCompare = (a: T, b: T) => a.calls - b.calls
  const deviceSelfDurationCompare = (a: T, b: T) =>
    (a.device_self_duration || 0) - (b.device_self_duration || 0)
  const deviceTotalDurationCompare = (a: T, b: T) =>
    (a.device_total_duration || 0) - (b.device_total_duration || 0)
  const hostSelfDurationCompare = (a: T, b: T) =>
    (a.host_self_duration || 0) - (b.host_self_duration || 0)
  const hostTotalDurationCompare = (a: T, b: T) =>
    (a.host_total_duration || 0) - (b.host_total_duration || 0)
  const tcEligibleCompare = (a: T, b: T) =>
    a.tc_eligible!.localeCompare(b.tc_eligible!)
  const tcSelfRatioCompare = (a: T, b: T) =>
    (a.tc_self_ratio || 0) - (b.tc_self_ratio || 0)
  const tcTotalRatioCompare = (a: T, b: T) =>
    (a.tc_total_ratio || 0) - (b.tc_total_ratio || 0)

  const columns: ColumnsType<T> = [
    {
      dataIndex: 'name',
      key: 'name',
      title: 'Name',
      sorter: nameCompare
    },
    hasInputShape
      ? {
          dataIndex: 'input_shape',
          key: 'input_shape',
          title: 'Input Shape'
        }
      : undefined,
    {
      dataIndex: 'calls',
      sorter: callsCompare,
      key: 'calls',
      title: 'Calls'
    },
    hasDeviceSelfDuration
      ? {
          dataIndex: 'device_self_duration',
          key: 'device_self_duration',
          title: 'Device Self Duration (us)',
          sorter: deviceSelfDurationCompare,
          // Use device_self_duration as default sort if defaultSort is unspecified
          defaultSortOrder: defaultSort ? undefined : ('descend' as const)
        }
      : undefined,
    hasDeviceTotalDuration
      ? {
          dataIndex: 'device_total_duration',
          key: 'device_total_duration',
          title: 'Device Total Duration (us)',
          sorter: deviceTotalDurationCompare
        }
      : undefined,
    {
      dataIndex: 'host_self_duration',
      key: 'host_self_duration',
      title: 'Host Self Duration (us)',
      sorter: hostSelfDurationCompare
    },
    {
      dataIndex: 'host_total_duration',
      key: 'host_total_duration',
      title: 'Host Total Duration (us)',
      sorter: hostTotalDurationCompare
    },
    hasTcEligible
      ? {
          dataIndex: 'tc_eligible',
          key: 'tc_eligible',
          title: 'Tensor Cores Eligible',
          sorter: tcEligibleCompare
        }
      : undefined,
    hasTcSelfRatio
      ? {
          dataIndex: 'tc_self_ratio',
          key: 'tc_self_ratio',
          title: 'Tensor Cores Self(%)',
          sorter: tcSelfRatioCompare
        }
      : undefined,
    hasTcTotalRatio
      ? {
          dataIndex: 'tc_total_ratio',
          key: 'tc_total_ratio',
          title: 'Tensor Cores Total(%)',
          sorter: tcTotalRatioCompare
        }
      : undefined
  ].filter(isDef)
  columns.forEach((column) => {
    if (column.key == defaultSort) {
      column.defaultSortOrder = 'descend' as const
    }
    if (tooltips[column.key as string]) {
      column.showSorterTooltip = {
        title: tooltips[column.key as string],
        overlayClassName: classes?.tooltip
      }
    }
  })
  return columns
}

let uid = 1
export function attachId<
  T extends CallStackTableDataInner | OperationTableDataInner
>(data: T[]): T[] {
  return data.map((d) => ({
    ...d,
    key: uid++
  }))
}
