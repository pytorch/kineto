/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export function isDef<T>(v: T | undefined | null): v is T {
  return v !== null && v !== undefined
}

export function assertDef<T>(v: T | undefined | null): asserts v is T {
  if (!isDef(v)) {
    throw new Error('Must be defined')
  }
}
