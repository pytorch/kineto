/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import { Table } from 'antd'
import * as React from 'react'
import { Graph } from '../../api'

interface IProps {
  graph: Graph
  sortColumn?: string
  initialPageSize?: number
  onRowSelected?: (record?: object, rowIndex?: number) => void
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap'
  },
  row: {
    wordBreak: 'break-word'
  }
}))

const getTableColumns = function (
  columns: any,
  sort: string | undefined,
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

const getTableRows = function (rows: any) {
  return rows.map(function (row: any) {
    let i = 0
    const res: any = {}
    row.forEach(function (entry: any) {
      res['col' + i++] = entry
    })
    return res
  })
}

export const AntTableChart: React.FC<IProps> = (props) => {
  const { graph, sortColumn, initialPageSize, onRowSelected } = props
  const classes = useStyles(props)

  const rows = React.useMemo(() => getTableRows(graph.rows), [graph.rows])

  const columns = React.useMemo(
    () => getTableColumns(graph.columns, sortColumn, classes.tooltip),
    [graph.columns, sortColumn, classes.tooltip]
  )

  // key is used to reset the Table state (page and sort) if the columns change
  const key = React.useMemo(() => Math.random() + '', [graph.columns])

  const [pageSize, setPageSize] = React.useState(initialPageSize ?? 30)
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size)
  }

  const onRow = (record: object, rowIndex?: number) => {
    return {
      onMouseEnter: (event: any) => {
        if (onRowSelected) {
          onRowSelected(record, rowIndex)
        }
      },
      onMouseLeave: (event: any) => {
        if (onRowSelected) {
          onRowSelected(undefined, undefined)
        }
      }
    }
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
      rowClassName={classes.row}
      key={key}
      onRow={onRow}
    />
  )
}
