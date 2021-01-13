/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import debounce from '@material-ui/core/utils/debounce'
import * as React from 'react'
import { value } from '.'
import * as api from '../api'

export function useSearch(
  searchName: string,
  columnName: string,
  table: api.Graph | undefined
): [api.Graph | undefined] {
  const [searchNameDebounce, setSearchNameDebounce] = React.useState(searchName)

  const onSearchOperatorNameChanged = React.useCallback(
    debounce((value: string) => {
      setSearchNameDebounce(value.trim())
    }, 500),
    []
  )

  React.useEffect(() => {
    onSearchOperatorNameChanged(searchName)
  }, [searchName])

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
