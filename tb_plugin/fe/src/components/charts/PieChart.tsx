/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { value } from '../../utils'
import { useResizeEventDependency } from '../../utils/resize'

interface IProps {
  graph: Graph
  height?: number
  top?: number
  noLegend?: boolean
  title?: string
  colors?: Array<string>
  tooltip_mode?: string
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: IProps) => props.height ?? 300
  }
}))

const noLegendArea = { left: '5%', width: '90%', top: '5%', height: '90%' }
const normalArea = { left: '5%', width: '95%' }
const noTitleArea = { left: '5%', width: '95%', top: '10%', height: '80%' }

export const PieChart: React.FC<IProps> = (props) => {
  const {
    graph,
    height = 300,
    top,
    noLegend,
    title,
    colors,
    tooltip_mode = 'both'
  } = props
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
      title,
      pieHole: 0.4,
      tooltip: { trigger: 'selection', isHtml: true, text: tooltip_mode },
      chartArea: noLegend ? noLegendArea : !title ? noTitleArea : normalArea,
      legend: noLegend ? 'none' : undefined,
      sliceVisibilityThreshold: 0,
      colors
    }

    const chart = new google.visualization.PieChart(element)

    google.visualization.events.addListener(
      chart,
      'onmouseover',
      function (entry: any) {
        chart.setSelection([{ row: entry.row }])
      }
    )

    google.visualization.events.addListener(chart, 'onmouseout', function () {
      chart.setSelection([])
    })

    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [graph, height, top, resizeEventDependency])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
