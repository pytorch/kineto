/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import * as api from '../api'

export interface IProps {
  run: string
  worker: string
  view: string
}

const useStyles = makeStyles(() => ({
  root: {
    flexGrow: 1
  },
  frame: {
    width: '100%',
    height: 'calc(100vh - 48px)',
    border: 'none'
  }
}))

export const TraceView: React.FC<IProps> = (props) => {
  const { run, worker, view } = props
  const classes = useStyles()

  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const [[traceData, resolveTraceData]] = React.useState(() => {
    let resolve: (v: string) => void = undefined!
    const pormise = new Promise<string>((r) => {
      resolve = r
    })
    return [pormise, resolve] as const
  })

  React.useEffect(() => {
    api.defaultApi.traceGet(run, worker, view).then((resp) => {
      resolveTraceData(JSON.stringify(resp))
    })
  }, [run, worker, view, resolveTraceData])

  React.useEffect(() => {
    function callback(event: MessageEvent) {
      const data = event.data || {}
      if (data.msg === 'ready') {
        traceData.then((data) => {
          iframeRef.current?.contentWindow?.postMessage(
            { msg: 'data', data },
            '*'
          )
        })
      }
    }

    window.addEventListener('message', callback)
    return () => {
      window.removeEventListener('message', callback)
    }
  }, [])

  return (
    <div className={classes.root}>
      {React.useMemo(
        () => (
          <iframe
            className={classes.frame}
            ref={iframeRef}
            src="/data/plugin/pytorch_profiler/trace_embedding.html"
          ></iframe>
        ),
        []
      )}
    </div>
  )
}
