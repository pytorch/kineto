/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { ValueAndFormat } from '../api'

export function firstOrUndefined<T>(v: T[] | undefined | null): T | undefined {
  if (!v || !v.length) return undefined
  return v[0]
}

export function sleep(delay: number) {
  return new Promise((resolve) => setTimeout(resolve, delay))
}

export function isValueAndFormat(v: any): v is ValueAndFormat {
  return 'f' in v && 'v' in v
}

export function value(
  v: boolean | number | string | ValueAndFormat
): boolean | number | string {
  return typeof v === 'object' && isValueAndFormat(v) ? v.v : v
}
