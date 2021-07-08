/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { useResizeEventDependency } from '../../utils/resize'

interface IProps {
  title?: string
  units?: string
  colors?: Array<string>
  chartData: ColumnChartData
}

const useStyles = makeStyles(() => ({
  root: {
    height: 500
  }
}))

export interface ColumnChartData {
  legends: Array<string>
  barLabels: Array<string>
  barHeights: Array<Array<number>>
}

export const ColumnChart: React.FC<IProps> = (props) => {
  const { title, units, colors, chartData } = props
  const { legends, barLabels, barHeights } = chartData
  const classes = useStyles()
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const data = new google.visualization.DataTable()
    data.addColumn({
      type: 'string',
      label: 'Worker'
    })
    legends.forEach((label) => {
      data.addColumn({
        type: 'number',
        label
      })
    })
    const rows = barHeights.map((heights, i) =>
      [barLabels[i] as string | number].concat(heights)
    )
    data.addRows(rows)

    const options = {
      height: 500,
      title,
      isStacked: true,
      legend: { position: 'bottom' },
      vAxis: {
        title: units
      },
      tooltip: { isHtml: true },
      chartArea: {
        left: '15%',
        width: '80%',
        top: title ? '10%' : '5%'
      },
      colors
    }

    const chart = new google.visualization.ColumnChart(element)

    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [title, chartData, resizeEventDependency])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
