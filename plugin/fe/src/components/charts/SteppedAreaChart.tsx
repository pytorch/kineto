/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'

interface IProps {
  graph: Graph
  height?: number
  hAxisTitle?: string
  vAxisTitle?: string
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height
  }
}))

export const SteppedAreaChart: React.FC<IProps> = (props) => {
  const { graph, height = 400, hAxisTitle, vAxisTitle } = props
  const classes = useStyles({ height })
  const graphRef = React.useRef<HTMLDivElement>(null)

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const data = new google.visualization.DataTable()
    graph.columns.forEach((column) => {
      data.addColumn(column.type, column.name)
    })
    data.addRows(graph.rows)

    const options = {
      title: graph.title,
      isStacked: true,
      height,
      legend: { position: 'bottom' },
      chartArea: { left: '15%', width: '80%', top: '10%' },
      hAxis: {
        title: hAxisTitle
      },
      vAxis: {
        title: vAxisTitle
      }
    }

    const chart = new google.visualization.SteppedAreaChart(element)

    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [graph, height])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
