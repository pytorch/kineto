/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import { CallStackTableData, OperationTableDataInner } from '../../api'
import { Table, TableProps } from 'antd'

import * as api from '../../api'
import { transformTableData, TransformedCallStackDataInner } from './transform'
import { attachId, getCommonOperationColumns } from './common'
import { OperationGroupBy } from '../../constants/groupBy'
import { makeExpandIcon } from './ExpandIcon'
import { CallFrameList } from './CallFrameList'

export interface IProps {
  data: OperationTableDataInner
  run: string
  worker: string
  span: string
  groupBy: OperationGroupBy
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap'
  }
}))

const expandIcon = makeExpandIcon<TransformedCallStackDataInner>(
  'View call frames',
  (record) => !record.callStackFrames.length
)

const rowExpandable = (record: TransformedCallStackDataInner) =>
  !!record.callStackFrames.length
const expandedRowRender = (record: TransformedCallStackDataInner) => (
  <CallFrameList callFrames={record.callStackFrames} />
)

export const CallStackTable = (props: IProps) => {
  const { data, run, worker, span, groupBy } = props
  const { name, input_shape } = data
  const classes = useStyles(props)

  const [stackData, setStackData] = React.useState<
    CallStackTableData | undefined
  >(undefined)
  const [tooltips, setTooltips] = React.useState<any | undefined>()

  React.useEffect(() => {
    api.defaultApi
      .operationStackGet(run, worker, span, groupBy, name, input_shape)
      .then((resp) => {
        setTooltips(resp.metadata.tooltips)
        setStackData(resp.data)
      })
  }, [name, input_shape, run, worker, span, groupBy])

  const transformedData = React.useMemo(
    () => stackData && transformTableData(attachId(stackData)),
    [stackData]
  )

  const columns = React.useMemo(
    () =>
      transformedData &&
      getCommonOperationColumns(transformedData, undefined, tooltips, classes),
    [transformedData]
  )

  const expandIconColumnIndex = columns?.length

  const expandable: TableProps<TransformedCallStackDataInner>['expandable'] = React.useMemo(
    () => ({
      expandIconColumnIndex,
      expandIcon,
      expandedRowRender,
      rowExpandable
    }),
    [expandIconColumnIndex]
  )

  return (
    <Table
      loading={!transformedData}
      size="small"
      bordered
      columns={columns}
      dataSource={transformedData}
      expandable={expandable}
    />
  )
}
