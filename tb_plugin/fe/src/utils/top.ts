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

  const [topText, setTopText] = React.useState(String(options.defaultTop ?? 15))
  const [actualTop, setActualTop] = React.useState<number | undefined>(
    Number(topText)
  )
  const [useTop, setUseTop] = React.useState(
    options.defaultUseTop ?? UseTop.NotUse
  )

  const setActualDebounce = !options.noDebounce
    ? React.useCallback(debounce(setActualTop, options.wait ?? 500), [])
    : setActualTop
  React.useEffect(() => {
    if (useTop !== UseTop.Use) {
      setActualDebounce(undefined)
    } else if (topIsValid(topText)) {
      setActualDebounce(Number(topText))
    } else {
      setActualDebounce(actualTop)
    }
  }, [topText, useTop])

  return [topText, actualTop, useTop, setTopText, setUseTop] as const
}

export function topIsValid(topText: string) {
  const top = Number(topText)
  return !Number.isNaN(top) && top > 0 && Number.isInteger(top)
}
