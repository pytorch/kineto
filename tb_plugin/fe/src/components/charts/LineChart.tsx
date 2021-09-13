/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { useResizeEventDependency } from '../../utils/resize'

interface IProps {
  graph: Graph
  device: string
  height?: number
  hAxisTitle?: string
  vAxisTitle?: string
  explorerOptions?: object
  onSelectionChanged?: (start: number, end: number) => void
  record?: object
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height
  }
}))

export const LineChart: React.FC<IProps> = (props) => {
  const {
    graph,
    device,
    height = 400,
    hAxisTitle,
    vAxisTitle,
    onSelectionChanged,
    explorerOptions,
    record
  } = props
  const classes = useStyles({ height })
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()
  const [chartObj, setChartObj] = React.useState<any | undefined>()

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

    const options = {
      title: graph.title,
      isStacked: true,
      height,
      legend: { position: 'bottom' },
      tooltip: { isHtml: true },
      hAxis: {
        title: hAxisTitle
      },
      vAxis: {
        title: vAxisTitle
      },
      explorer: explorerOptions
    }

    const chart = new google.visualization.LineChart(element)

    // Disable selection of single point
    google.visualization.events.addListener(chart, 'select', function () {
      chart.setSelection()
    })

    google.visualization.events.addListener(chart, 'ready', function () {
      // console.log('ready')
      var zoomLast = getCoords()
      var observer = new MutationObserver(function () {
        var zoomCurrent = getCoords()
        if (JSON.stringify(zoomLast) !== JSON.stringify(zoomCurrent)) {
          zoomLast = getCoords()
          if (onSelectionChanged) {
            onSelectionChanged(zoomLast.x_min, zoomLast.x_max)
          }
        }
      })
      if (graphRef.current) {
        observer.observe(graphRef.current, {
          childList: true,
          subtree: true
        })
      }
    })

    function getCoords() {
      var chartLayout = chart.getChartLayoutInterface()
      var chartBounds = chartLayout.getChartAreaBoundingBox()

      return {
        x_min: chartLayout.getHAxisValue(chartBounds.left),
        x_max: chartLayout.getHAxisValue(chartBounds.width + chartBounds.left)
      }
    }

    chart.draw(data, options)
    setChartObj(chart)
  }, [device, height, resizeEventDependency])

  React.useEffect(() => {
    console.log(JSON.stringify(record) + ' aaaaaaaa is selected')
    if (chartObj) {
      const selected = chartObj.getSelection()
      console.log(chartObj)
      console.log(selected)
    }
  }, [record, chartObj])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  )
}
