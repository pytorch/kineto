/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export function navToCode(filename: string, line: number) {
  window.parent.parent.postMessage(
    {
      filename,
      line
    },
    '*'
  )
}
