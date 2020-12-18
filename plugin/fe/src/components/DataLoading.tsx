/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { FullCircularProgress } from './FullCircularProgress'

interface IProps<T> {
  value: T | undefined | null
  children: (t: T) => JSX.Element
}

export function DataLoading<T>(props: IProps<T>) {
  if (props.value === undefined || props.value === null) {
    return <FullCircularProgress />
  }

  return props.children(props.value)
}
