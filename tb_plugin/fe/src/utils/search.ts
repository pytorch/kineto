/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { value } from '.'
import * as api from '../api'
import { useDebounce } from './debounce'

export function useSearch(
  searchName: string,
  columnName: string,
  table: api.Graph | undefined
): [api.Graph | undefined] {
  const searchNameDebounce = useDebounce(searchName.trim(), 500)

  const searchedTable: api.Graph | undefined = React.useMemo(() => {
    if (!searchNameDebounce) {
      return table
    }

    if (!table) {
      return undefined
    }

    const columnNameToFind = columnName.toLowerCase()
    const nameColumnIdx = table.columns.findIndex(
      (c) => c.name.toLowerCase() === columnNameToFind
    )
    if (nameColumnIdx < 0) {
      return table
    }

    return {
      ...table,
      rows: table.rows.filter((x) => {
        const cell = value(x[nameColumnIdx])
        return typeof cell === 'string' && cell.includes(searchNameDebounce)
      })
    }
  }, [table, searchNameDebounce])
  return [searchedTable]
}

export function useSearchDirectly<T>(
  searchName: string,
  field: (v: T) => string,
  table: T[] | undefined
): [T[] | undefined] {
  const searchNameDebounce = useDebounce(searchName.trim(), 500)

  const result = React.useMemo(() => {
    if (!searchNameDebounce) {
      return table
    }

    if (!table) {
      return undefined
    }

    return table.filter((row) => {
      return field(row).toLowerCase().includes(searchNameDebounce.toLowerCase())
    })
  }, [table, field, searchNameDebounce])
  return [result]
}
