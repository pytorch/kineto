/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export function navToCode(filename: string, line: number) {
  console.log(filename, line)
  window.parent.parent.postMessage(
    {
      filename,
      line
    },
    '*'
  )
}
