/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { GraphicEqSharp } from '@material-ui/icons'
import { format } from 'url'
import * as api from '../api'
import { assertDef, isDef } from '../utils/def'

export function transformPerformanceIntoTable(
  performances: api.Performance[], unit: string) : api.Graph {

  const rows: api.Graph['rows'] = []
  const queue = [...performances]
  while (queue.length) {
    const first = queue.shift()
    assertDef(first)

    const row: api.Graph['rows'][number] = []
    const { name, value, extra, children } = first
    assertDef(value)
    assertDef(extra)

    row.push(name)
    row.push(parseFloat(value))
    row.push(extra)

    if (isDef(children) && children.length) {
      queue.push(...children)
    }
    rows.push(row)
  }
  const columns: api.GraphColumn[] = [
    { type: 'string', name: 'Category' },
    { type: 'number', name: `Time Duration (${unit}s)` },
    { type: 'number', name: 'Percentage (%)' }
  ]
  return {
    columns,
    rows
  }
}

export function transformPerformanceIntoPie(performances: api.Performance[]) {
  const columns: api.GraphColumn[] = [
    { type: 'string', name: 'Name' },
    { type: 'number', name: 'Value' }
  ]

  const rows: api.Graph['rows'] = []
  const queue: api.Performance[] = []
  performances.forEach((topLevel) => {
    if (topLevel.children) {
      queue.push(...topLevel.children)
    }
  })

  while (queue.length) {
    const first = queue.shift()
    assertDef(first)

    const row: api.Graph['rows'][number] = []
    const { name, value, children } = first
    assertDef(value)

    row.push(name)
    row.push(Number.parseInt(value, 10))

    if (isDef(children) && children.length) {
      queue.push(...children)
    }

    rows.push(row)
  }

  return {
    columns,
    rows
  }
}

export function transformDictIntoTooltipHtml(graph: api.Graph, unit: string) {
  if (graph == undefined) {
    return
  }
  graph.rows.forEach(row => {
    for (let i = 1; i < row.length; i++) {
      if (i % 2 == 0) {
        var row_json = JSON.parse(JSON.stringify(row[i]))
        const total = unit == "m"? Number(usToMs(row_json.Total)): row_json.Total
        const partCost = unit == "m"? Number(usToMs(row_json.PartCost)): row_json.PartCost

        const format_str = `<div class="visualization-tooltip" style="white-space: nowrap;">
              Step ${row_json.Step}<br>
              Total: ${total}${unit}s<br>      
              <b>${row_json.PartName}: ${partCost}${unit}s</b><br>
              Percentage: ${row_json.Percentage}%
              </div>`
        row[i] = format_str
      } else {
        row[i] = unit == "m"? Number(usToMs(row[i] as string)): row[i]
      }
    }
  });
  
}

export function autoScalePerformanceData(performances: api.Performance[]): boolean {
  // Adjust units
  const queue = [...performances]
  var isScaled:boolean = false
  while (queue.length) {
    const first = queue.shift()
    assertDef(first)
    var value = first.value
    var children = first.children
    assertDef(value)
    
    if (Number(value) > 10000) {
      first.value = usToMs(value)
      isScaled = true
    }

    if (isDef(children) && children.length) {
      if (isScaled) {
        children.forEach((i) => {
          assertDef(i.value)
          i.value = usToMs(i.value)
        })
      }
      queue.push(...children)
    }
  }
  return isScaled
}

function usToMs(us: string): string {
  return (Number(us)/1000).toFixed(2)
}