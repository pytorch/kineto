/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export type Arguments<T extends (...args: any[]) => void> = T extends (
  ...args: infer A
) => void
  ? A
  : never
