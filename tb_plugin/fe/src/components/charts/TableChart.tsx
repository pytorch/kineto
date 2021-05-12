/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { useResizeEventDependency } from '../../utils/resize'

interface IProps {
  graph: Graph
  sortColumn?: number
  height?: number
  allowHtml?: boolean
  setCellProperty?: (
    row: number,
    column: number,
    cb: (key: string, value: any) => void
  ) => void
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: IProps) => props.height
  }
}))

export const TableChart: React.FC<IProps> = (props) => {
  const { graph, sortColumn, setCellProperty, allowHtml } = props
  const classes = useStyles(props)
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const data = new google.visualization.DataTable()
    graph.columns.forEach((column) => {
      data.addColumn({
        type: column.type,
        label: column.name,
        role: column.role,
        p: column.p
      })
    })
    data.addRows(graph.rows)

    if (setCellProperty) {
      for (let row = 0; row < graph.rows.length; ++row) {
        for (let column = 0; column < graph.columns.length; ++column) {
          setCellProperty(row, column, (key: string, value: any) => {
            data.setProperty(row, column, key, value)
          })
        }
      }
    }

    const options = {
      width: '100%',
      height: '100%',
      page: 'enable',
      allowHtml,
      pageSize: 30,
      tooltip: { isHtml: true },
      sortColumn: sortColumn,
      sortAscending: false
    }

    const chart = new google.visualization.Table(element)

    /* `chart.draw()` removes the contents of `element` and rebuilds it. This can cause a jump in the scroll position
     * if the height/width change to 0. Since we can't change the code of Google Charts, we temporarily lock the dims
     * of the parent container. */
    if (element.offsetHeight > 0) {
      element.parentElement!.style.height = element.offsetHeight + 'px'
    }
    chart.draw(data, options)
    element.parentElement!.style.height = ''
  }, [graph, resizeEventDependency])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
