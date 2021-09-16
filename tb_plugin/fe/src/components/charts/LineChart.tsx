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
  vAxisTitle?: string
  initialSelectionStart?: number
  initialSelectionEnd?: number
  onSelectionChanged?: (start: number, end: number) => void
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height
  }
}))

export const LineChart: React.FC<IProps> = (props) => {
  const {
    graph,
    height = 400,
    hAxisTitle,
    vAxisTitle,
    initialSelectionStart,
    initialSelectionEnd,
    onSelectionChanged
  } = props
  const classes = useStyles({ height })
  const graphRef = React.useRef<HTMLDivElement>(null)
  const rangeDivRef = React.useRef<HTMLDivElement>(null)
  const selectedDivRef = React.useRef<HTMLDivElement>(null)
  const initialSelectionDivRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    const rangeDiv = rangeDivRef.current
    const selectedDiv = selectedDivRef.current
    if (!element || !rangeDiv || !selectedDiv) return

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
      }
    }

    const chart = new google.visualization.LineChart(element)

    chart.draw(data, options)
    const cli = chart.getChartLayoutInterface()
    const graphRect = cli.getChartAreaBoundingBox()
    const rect = element.getBoundingClientRect()

    const state = {
      startX: initialSelectionStart
        ? cli.getXLocation(initialSelectionStart) - graphRect.left
        : 0,
      endX: initialSelectionEnd
        ? cli.getXLocation(initialSelectionEnd) - graphRect.left
        : 0,
      mouseDown: false,
      minX: 0,
      maxX: 0
    }

    rangeDiv.style.width = graphRect.width + 'px'
    rangeDiv.style.height = graphRect.height + 'px'
    selectedDiv.style.height = graphRect.height + 'px'

    function clamp(x: number) {
      return Math.max(0, Math.min(x, graphRect.width))
    }

    const renderSelection = () => {
      state.minX = clamp(Math.min(state.startX, state.endX))
      state.maxX = clamp(Math.max(state.startX, state.endX))
      selectedDiv.style.width = state.maxX - state.minX + 'px'
      selectedDiv.style.marginLeft = state.minX + 'px'
    }
    renderSelection()

    const onMouseUp = (event: MouseEvent) => {
      const rect = rangeDiv?.getBoundingClientRect()
      if (rect && state.mouseDown) {
        state.endX = event.clientX - rect.x
        state.mouseDown = false
        renderSelection()
        if (onSelectionChanged) {
          const startVal = cli.getHAxisValue(state.minX + graphRect.left)
          const endVal = cli.getHAxisValue(state.maxX + graphRect.left)
          onSelectionChanged(startVal, endVal)
        }
      }
      window.removeEventListener('mouseup', onMouseUp)
      element.removeEventListener('mouseleave', onMouseUp)
    }

    const onMouseMove = (event: MouseEvent) => {
      const rect = rangeDiv?.getBoundingClientRect()
      if (rect && state.mouseDown) {
        state.endX = event.clientX - rect.x
        renderSelection()
      } else {
        window.removeEventListener('mousemove', onMouseMove)
      }
    }

    const onMouseDown = (event: MouseEvent) => {
      const rect = rangeDiv?.getBoundingClientRect()
      if (rect) {
        event.preventDefault()
        state.startX = event.clientX - rect.x
        state.endX = event.clientX - rect.x
        state.mouseDown = true
        renderSelection()
        window.addEventListener('mousemove', onMouseMove)
        window.addEventListener('mouseup', onMouseUp)
        element.addEventListener('mouseleave', onMouseUp)
      }
    }
    rangeDiv.addEventListener('mousedown', onMouseDown)

    rangeDiv.style.marginTop = -(rect.height - graphRect.top) + 'px'
    rangeDiv.style.marginLeft = graphRect.left + 'px'

    return () => {
      chart.clearChart()
      rangeDiv.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      element.removeEventListener('mouseleave', onMouseUp)
    }
  }, [graph, height, resizeEventDependency])

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
      <div
        ref={rangeDivRef}
        style={{
          backgroundColor: 'transparent',
          position: 'relative',
          cursor: 'crosshair'
        }}
      >
        <div
          ref={selectedDivRef}
          style={{
            backgroundColor: 'red',
            marginLeft: 0,
            width: 0,
            opacity: 0.2
          }}
        ></div>
      </div>
    </div>
  )
}
