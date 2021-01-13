/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import debounce from '@material-ui/core/utils/debounce'
import * as React from 'react'

export enum UseTop {
  NotUse = 'NotUse',
  Use = 'Use'
}

interface IOptions {
  defaultTop?: number
  defaultUseTop?: UseTop
  noDebounce?: boolean
  wait?: number
}

export function useTopN(options?: IOptions) {
  options ??= {}

  const [top, setTop] = React.useState(options.defaultTop ?? 15)
  const [actualTop, setActualTop] = React.useState<number | undefined>(top)
  const [useTop, setUseTop] = React.useState(
    options.defaultUseTop ?? UseTop.NotUse
  )

  const setActualDebounce = !options.noDebounce
    ? React.useCallback(debounce(setActualTop, options.wait ?? 500), [])
    : setActualTop
  React.useEffect(() => {
    setActualDebounce(useTop === UseTop.Use && top > 0 ? top : undefined)
  }, [top, useTop])

  return [top, actualTop, useTop, setTop, setUseTop] as const
}
