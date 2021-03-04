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

  const [traceData, setTraceData] = React.useState<Promise<string> | null>(null)
  const [traceViewReady, setTraceViewReady] = React.useState(false)

  React.useEffect(() => {
    setTraceData(
      api.defaultApi.traceGet(run, worker, view).then((resp) => {
        return JSON.stringify(resp)
      })
    )
  }, [run, worker, view])

  React.useEffect(() => {
    function callback(event: MessageEvent) {
      const data = event.data || {}
      if (data.msg === 'ready') {
        setTraceViewReady(true)
      }
    }

    window.addEventListener('message', callback)
    return () => {
      window.removeEventListener('message', callback)
    }
  }, [])

  React.useEffect(() => {
    if (traceData && traceViewReady) {
      traceData.then((data) => {
        iframeRef.current?.contentWindow?.postMessage(
          { msg: 'data', data },
          '*'
        )
      })
    }
  }, [traceData, traceViewReady])

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
