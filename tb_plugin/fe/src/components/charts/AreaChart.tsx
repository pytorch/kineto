/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { useResizeEventDependency } from '../../utils/resize'

interface IProps {
  graph: Graph
  height?: number
  hAxisTitle?: string
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height
  }
}))

export const AreaChart: React.FC<IProps> = (props) => {
  const { graph, height = 400, hAxisTitle } = props
  const classes = useStyles({ height })
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const data = new google.visualization.DataTable()
    data.addColumn('string', 'step')
    graph.columns.forEach((column) => {
      data.addColumn({
        type: column.type,
        label: column.name,
        role: column.role,
        p: column.p
      })
    })
    data.addRows(graph.rows.map((x, i) => [(i + 1).toString(), ...x]))

    const options = {
      title: graph.title,
      isStacked: true,
      height,
      legend: { position: 'bottom' },
      tooltip: { isHtml: true },
      chartArea: { left: '15%', width: '80%', top: '10%' },
      hAxis: {
        title: hAxisTitle
      }
    }

    const chart = new google.visualization.AreaChart(element)

    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [graph, height, resizeEventDependency])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
