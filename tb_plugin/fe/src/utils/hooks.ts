/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'

const cbs: (() => void)[] = []
export const useOnResize = (cb: () => void) => {
  React.useEffect(() => {
    if (cbs.length === 0) {
      window.addEventListener('resize', () => {
        cbs.forEach((cb) => cb())
      })
    }
    cbs.push(cb)

    return () => {
      const idx = cbs.findIndex(cb)
      if (idx > -1) {
        cbs.splice(idx, 1)
      }
      if (cbs.length === 0) {
        window.removeEventListener('reset', cb)
      }
    }
  }, [cb])
}
