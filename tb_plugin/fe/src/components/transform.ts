/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as api from '../api'
import { assertDef, isDef } from '../utils/def'

export function transformPerformanceIntoTable(
  performances: api.Performance[]
): api.Graph {
  const columns: api.GraphColumn[] = [
    { type: 'string', name: 'Category' },
    { type: 'number', name: 'Time Duration (us)' },
    { type: 'number', name: 'Percentage (%)' }
  ]

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
    row.push(value)
    row.push(extra)

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
