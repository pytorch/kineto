/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { value } from '../../utils'

interface IProps {
  graph: Graph
  height?: number
  top?: number
  noLegend?: boolean
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: IProps) => props.height ?? 300
  }
}))

const noLegendArea = { left: '5%', width: '90%', top: '5%', height: '90%' }
const normalArea = { left: '5%', width: '95%' }

export const PieChart: React.FC<IProps> = (props) => {
  const { graph, height = 300, top, noLegend } = props
  const classes = useStyles(props)
  const graphRef = React.useRef<HTMLDivElement>(null)

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

    const rows =
      top === undefined
        ? graph.rows
        : graph.rows
            .sort((a, b) => (value(b[1]) as number) - (value(a[1]) as number))
            .slice(0, top)
    data.addRows(rows)

    const options = {
      height,
      width: '100%',
      title: graph.title,
      pieHole: 0.4,
      tooltip: { trigger: 'selection', isHtml: true },
      chartArea: noLegend ? noLegendArea : normalArea,
      legend: noLegend ? 'none' : undefined,
      sliceVisibilityThreshold: 0
    }

    const chart = new google.visualization.PieChart(element)

    google.visualization.events.addListener(chart, 'onmouseover', function (
      entry: any
    ) {
      chart.setSelection([{ row: entry.row }])
    })

    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [graph, height, top])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
