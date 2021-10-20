/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { Table } from 'antd'
import { makeStyles } from '@material-ui/core'

export interface IProps {
  data: any
  sort: string
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap'
  }
}))

const getMemoryStatsTableColumns = function (
  columns: any,
  sort: string,
  tooltipClass: string
) {
  let i = 0
  return columns.map(function (col: any) {
    const key = 'col' + i++
    const stringCompare = (a: any, b: any) => a[key].localeCompare(b[key])
    const numberCompare = (a: any, b: any) => (a[key] || 0) - (b[key] || 0)
    return {
      dataIndex: key,
      key: key,
      title: col.name,
      sorter: col.type == 'string' ? stringCompare : numberCompare,
      defaultSortOrder: sort == col.name ? ('descend' as const) : undefined,
      showSorterTooltip: col.tooltip
        ? { title: col.tooltip, overlayClassName: tooltipClass }
        : true
    }
  })
}

const getMemoryStatsTableRows = function (rows: any) {
  return rows.map(function (row: any) {
    let i = 0
    const res: any = {}
    row.forEach(function (entry: any) {
      res['col' + i++] = entry
    })
    return res
  })
}

export const MemoryStatsTable = (props: IProps) => {
  const { data, sort } = props
  const classes = useStyles()

  const rows = React.useMemo(() => getMemoryStatsTableRows(data.rows), [
    data.rows
  ])

  const columns = React.useMemo(
    () => getMemoryStatsTableColumns(data.columns, sort, classes.tooltip),
    [data.columns, sort, classes.tooltip]
  )

  const [pageSize, setPageSize] = React.useState(30)
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size)
  }

  return (
    <Table
      size="small"
      bordered
      columns={columns}
      dataSource={rows}
      pagination={{
        pageSize,
        pageSizeOptions: ['10', '20', '30', '50', '100'],
        onShowSizeChange
      }}
    />
  )
}
