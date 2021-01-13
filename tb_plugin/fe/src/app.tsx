/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import CssBaseline from '@material-ui/core/CssBaseline'
import Drawer from '@material-ui/core/Drawer'
import FormControl from '@material-ui/core/FormControl'
import ListSubheader from '@material-ui/core/ListSubheader'
import { makeStyles } from '@material-ui/core/styles'
import MenuItem from '@material-ui/core/MenuItem'
import Select, { SelectProps } from '@material-ui/core/Select'
import { Overview } from './components/Overview'
import * as React from 'react'
import { Operator } from './components/Operator'
import { Kernel } from './components/Kernel'
import * as api from './api'
import { firstOrUndefined } from './utils'
import { setup } from './setup'
import './styles.css'
import { TraceView } from './components/TraceView'
import { FullCircularProgress } from './components/FullCircularProgress'

export enum Views {
  Overview = 'Overview',
  Operator = 'Operator',
  Kernel = 'Kernel',
  Trace = 'Trace'
}

const ViewNames = {
  [Views.Overview]: Views.Overview,
  [Views.Operator]: Views.Operator,
  [Views.Kernel]: 'GPU Kernel',
  [Views.Trace]: Views.Trace
}

const drawerWidth = 340
const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex'
  },
  appBar: {
    width: `calc(100% - ${drawerWidth}px)`,
    marginLeft: drawerWidth
  },
  drawer: {
    width: drawerWidth,
    flexShrink: 0
  },
  drawerPaper: {
    width: drawerWidth,
    paddingLeft: theme.spacing(1),
    paddingRight: theme.spacing(1)
  },
  // necessary for content to be below app bar
  toolbar: theme.mixins.toolbar,
  content: {
    flexGrow: 1,
    backgroundColor: theme.palette.background.default,
    padding: theme.spacing(3)
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120
  }
}))

export const App = () => {
  const classes = useStyles()

  const [run, setRun] = React.useState<string>('')
  const [runs, setRuns] = React.useState<string[]>([])

  const [workers, setWorkers] = React.useState<string[]>([])
  const [worker, setWorker] = React.useState<string>('')

  const [views, setViews] = React.useState<Views[]>([])
  const [view, setView] = React.useState<Views | ''>('')
  const [loaded, setLoaded] = React.useState(false)

  React.useEffect(() => {
    setup().then(() => {
      setLoaded(true)
    })
  }, [])

  React.useEffect(() => {
    api.defaultApi.runsGet().then((runs) => {
      setRuns(runs)
    })
  }, [])

  React.useEffect(() => {
    setRun(firstOrUndefined(runs) ?? '')
  }, [runs])

  React.useEffect(() => {
    if (run) {
      api.defaultApi.workersGet(run).then((workers) => {
        setWorkers(workers)
      })
    }
  }, [run])

  React.useEffect(() => {
    setWorker(firstOrUndefined(workers) ?? '')
  }, [workers])

  React.useEffect(() => {
    if (run) {
      api.defaultApi.viewsGet(run).then((rawViews) => {
        const views = rawViews
          .map((v) => Views[Views[v as Views]])
          .filter(Boolean)
        setViews(views)
      })
    }
  }, [run])

  React.useEffect(() => {
    setView(firstOrUndefined(views) ?? '')
  }, [views])

  const handleRunChange: SelectProps['onChange'] = (event) => {
    setRun(event.target.value as string)
    setWorker('')
    setView('')
  }

  const handleWorkerChange: SelectProps['onChange'] = (event) => {
    setWorker(event.target.value as string)
  }

  const handleViewChange: SelectProps['onChange'] = (event) => {
    setView(event.target.value as Views)
  }

  const renderContent = () => {
    if (!loaded || !run || !worker || !view) {
      return <FullCircularProgress />
    }

    switch (view) {
      case Views.Overview:
        return <Overview run={run} worker={worker} view={view} />
      case Views.Operator:
        return <Operator run={run} worker={worker} view={view} />
      case Views.Kernel:
        return <Kernel run={run} worker={worker} view={view} />
      case Views.Trace:
        return <TraceView run={run} worker={worker} view={view} />
    }
  }

  return (
    <div className={classes.root}>
      <CssBaseline />
      <Drawer
        className={classes.drawer}
        variant="permanent"
        anchor="left"
        classes={{
          paper: classes.drawerPaper
        }}
      >
        <div className={classes.toolbar} />
        <ListSubheader>Runs</ListSubheader>
        <FormControl variant="outlined" className={classes.formControl}>
          <Select value={run} onChange={handleRunChange}>
            {runs.map((run) => (
              <MenuItem value={run}>{run}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <ListSubheader>Workers</ListSubheader>
        <FormControl variant="outlined" className={classes.formControl}>
          <Select value={worker} onChange={handleWorkerChange}>
            {workers.map((worker) => (
              <MenuItem value={worker}>{worker}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <ListSubheader>Views</ListSubheader>
        <FormControl variant="outlined" className={classes.formControl}>
          <Select value={view} onChange={handleViewChange}>
            {views.map((view) => (
              <MenuItem value={view}>{ViewNames[view]}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Drawer>
      <main className={classes.content}>{renderContent()}</main>
    </div>
  )
}
