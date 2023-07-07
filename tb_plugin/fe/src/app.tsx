/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Box from '@material-ui/core/Box'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import ClickAwayListener from '@material-ui/core/ClickAwayListener'
import CssBaseline from '@material-ui/core/CssBaseline'
import Divider from '@material-ui/core/Divider'
import Drawer from '@material-ui/core/Drawer'
import Fab from '@material-ui/core/Fab'
import FormControl from '@material-ui/core/FormControl'
import IconButton from '@material-ui/core/IconButton'
import ListSubheader from '@material-ui/core/ListSubheader'
import MenuItem from '@material-ui/core/MenuItem'
import Select, { SelectProps } from '@material-ui/core/Select'
import { makeStyles } from '@material-ui/core/styles'
import Tab from '@material-ui/core/Tab'
import Tabs from '@material-ui/core/Tabs'
import Typography from '@material-ui/core/Typography'
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'
import ChevronRightIcon from '@material-ui/icons/ChevronRight'
import 'antd/es/button/style/css'
import 'antd/es/list/style/css'
import 'antd/es/table/style/css'
import clsx from 'clsx'
import * as React from 'react'
import * as api from './api'
import { DiffOverview } from './components/DiffOverview'
import { DistributedView } from './components/DistributedView'
import { FullCircularProgress } from './components/FullCircularProgress'
import { Kernel } from './components/Kernel'
import { MemoryView } from './components/MemoryView'
import { ModuleView } from './components/ModuleView'
import { Operator } from './components/Operator'
import { Overview } from './components/Overview'
import { TraceView } from './components/TraceView'
import { setup } from './setup'
import './styles.css'
import { firstOrUndefined, sleep } from './utils'

export enum Views {
  Overview = 'Overview',
  Operator = 'Operator',
  Kernel = 'Kernel',
  Trace = 'Trace',
  Distributed = 'Distributed',
  Memory = 'Memory',
  Module = 'Module',
  Lightning = 'Lightning'
}

const ViewNames = {
  [Views.Overview]: Views.Overview,
  [Views.Operator]: Views.Operator,
  [Views.Kernel]: 'GPU Kernel',
  [Views.Trace]: Views.Trace,
  [Views.Distributed]: Views.Distributed,
  [Views.Memory]: Views.Memory,
  [Views.Module]: Views.Module,
  [Views.Lightning]: Views.Lightning
}

const drawerWidth = 340
const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex'
  },
  appBar: {
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen
    })
  },
  appBarShift: {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen
    })
  },
  menuButton: {
    marginRight: 36
  },
  hide: {
    display: 'none'
  },
  drawer: {
    width: drawerWidth,
    flexShrink: 0,
    whiteSpace: 'nowrap'
  },
  drawerOpen: {
    width: drawerWidth,
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen
    })
  },
  drawerClose: {
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen
    }),
    overflowX: 'hidden',
    width: 0,
    [theme.breakpoints.up('sm')]: {
      width: 0
    }
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    padding: theme.spacing(0, 1),
    // necessary for content to be below app bar
    ...theme.mixins.toolbar
  },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3)
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120
  },
  fab: {
    marginLeft: theme.spacing(1),
    marginTop: theme.spacing(1),
    position: 'absolute'
  },
  iconButton: {
    padding: '8px'
  }
}))

export const App = () => {
  const classes = useStyles()

  // #region - State

  const [selectedTab, setSelectedTab] = React.useState(0)

  const [run, setRun] = React.useState<string>('')
  const [runs, setRuns] = React.useState<string[]>([])
  const [runsLoading, setRunsLoading] = React.useState(true)

  const [workers, setWorkers] = React.useState<string[]>([])
  const [worker, setWorker] = React.useState<string>('')

  const [spans, setSpans] = React.useState<string[]>([])
  const [span, setSpan] = React.useState<string | ''>('')

  const [views, setViews] = React.useState<Views[]>([])
  const [view, setView] = React.useState<Views | ''>('')
  const [loaded, setLoaded] = React.useState(false)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const [diffLeftWorkerOptions, setDiffLeftWorkerOptions] = React.useState<
    string[]
  >([])
  const [diffLeftSpansOptions, setDiffLeftSpansOptions] = React.useState<
    string[]
  >([])
  const [diffLeftRun, setDiffLeftRun] = React.useState<string>('')
  const [diffLeftWorker, setDiffLeftWorker] = React.useState<string>('')
  const [diffLeftSpan, setDiffLeftSpan] = React.useState<string | ''>('')

  const [diffRightWorkerOptions, setDiffRightWorkerOptions] = React.useState<
    string[]
  >([])
  const [diffRightSpansOptions, setDiffRightSpansOptions] = React.useState<
    string[]
  >([])
  const [diffRightRun, setDiffRightRun] = React.useState<string>('')
  const [diffRightWorker, setDiffRightWorker] = React.useState<string>('')
  const [diffRightSpan, setDiffRightSpan] = React.useState<string | ''>('')

  const [open, setOpen] = React.useState(true)

  // #endregion

  React.useEffect(() => {
    setup().then(() => {
      setLoaded(true)
    })
  }, [])

  const continuouslyFetchRuns = async () => {
    while (true) {
      try {
        const runs = await api.defaultApi.runsGet()
        setRuns(runs.runs)
        setRunsLoading(runs.loading)
      } catch (e) {
        console.info('Cannot fetch runs: ', e)
      }
      await sleep(5000)
    }
  }

  React.useEffect(() => {
    continuouslyFetchRuns()
  }, [])

  React.useEffect(() => {
    if (!run || !runs.includes(run)) {
      setRun(firstOrUndefined(runs) ?? '')
    }
  }, [runs])

  // #region - Diff Left

  React.useEffect(() => {
    if (diffLeftRun) {
      api.defaultApi.workersGet(diffLeftRun, Views.Overview).then((workers) => {
        setDiffLeftWorkerOptions(workers)
      })
    }
  }, [diffLeftRun])

  React.useEffect(() => {
    if (diffLeftRun && diffLeftWorker) {
      api.defaultApi.spansGet(diffLeftRun, diffLeftWorker).then((spans) => {
        setDiffLeftSpansOptions(spans)
      })
    }
  }, [diffLeftRun, diffLeftWorker])

  // #endregion

  // #region - Diff Right

  React.useEffect(() => {
    if (diffRightRun) {
      api.defaultApi
        .workersGet(diffRightRun, Views.Overview)
        .then((workers) => {
          setDiffRightWorkerOptions(workers)
        })
    }
  }, [diffRightRun])

  React.useEffect(() => {
    if (diffRightRun && diffRightWorker) {
      api.defaultApi.spansGet(diffRightRun, diffRightWorker).then((spans) => {
        setDiffRightSpansOptions(spans)
      })
    }
  }, [diffRightRun, diffRightWorker])

  // #endregion

  // #region - normal

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

  React.useEffect(() => {
    if (run && view) {
      api.defaultApi.workersGet(run, view).then((workers) => {
        setWorkers(workers)
      })
    }
  }, [run, view])

  React.useEffect(() => {
    setWorker(firstOrUndefined(workers) ?? '')
  }, [workers])

  React.useEffect(() => {
    if (run && worker) {
      api.defaultApi.spansGet(run, worker).then((spans) => {
        setSpans(spans)
      })
    }
  }, [run, worker])

  React.useEffect(() => {
    setSpan(firstOrUndefined(spans) ?? '')
  }, [spans])

  // #endregion

  // #region - Event Handler
  const handleTabChange = (event: React.ChangeEvent<{}>, value: any) => {
    setSelectedTab(value as number)
  }

  const handleRunChange: SelectProps['onChange'] = (event) => {
    setRun(event.target.value as string)
    setView('')
    setWorker('')
    setSpan('')
  }

  const handleViewChange: SelectProps['onChange'] = (event) => {
    setView(event.target.value as Views)
    setWorker('')
    setSpan('')
  }

  const handleWorkerChange: SelectProps['onChange'] = (event) => {
    setWorker(event.target.value as string)
    setSpan('')
  }

  const handleSpanChange: SelectProps['onChange'] = (event) => {
    setSpan(event.target.value as string)
  }

  const handleDiffLeftRunChange: SelectProps['onChange'] = (event) => {
    setDiffLeftRun(event.target.value as string)
    setDiffLeftWorker('')
    setDiffLeftSpan('')
  }

  const handleDiffLeftWorkerChange: SelectProps['onChange'] = (event) => {
    setDiffLeftWorker(event.target.value as string)
    setDiffLeftSpan('')
  }

  const handleDiffLeftSpanChange: SelectProps['onChange'] = (event) => {
    setDiffLeftSpan(event.target.value as string)
  }

  const handleDiffRightRunChange: SelectProps['onChange'] = (event) => {
    setDiffRightRun(event.target.value as string)
    setDiffRightWorker('')
    setDiffRightSpan('')
  }

  const handleDiffRightWorkerChange: SelectProps['onChange'] = (event) => {
    setDiffRightWorker(event.target.value as string)
    setDiffRightSpan('')
  }

  const handleDiffRightSpanChange: SelectProps['onChange'] = (event) => {
    setDiffRightSpan(event.target.value as string)
  }

  const handleDrawerOpen = () => {
    setOpen(true)
    SetIframeActive()
  }

  const handleDrawerClose = () => {
    setOpen(false)
    SetIframeActive()
  }

  const SetIframeActive = () => {
    iframeRef.current?.focus()
  }

  // #endregion

  const renderContent = () => {
    if (!runsLoading && runs.length == 0) {
      return (
        <Card variant="outlined">
          <CardHeader title="No Runs Found"></CardHeader>
          <CardContent>
            <Typography>There are not any runs in the log folder.</Typography>
          </CardContent>
        </Card>
      )
    }

    if (!loaded || !run || !worker || !view || !span) {
      return <FullCircularProgress />
    }

    if (selectedTab === 0) {
      switch (view) {
        case Views.Overview:
          return <Overview run={run} worker={worker} span={span} />
        case Views.Operator:
          return <Operator run={run} worker={worker} span={span} />
        case Views.Kernel:
          return <Kernel run={run} worker={worker} span={span} />
        case Views.Trace:
          return (
            <TraceView
              run={run}
              worker={worker}
              span={span}
              iframeRef={iframeRef}
            />
          )
        case Views.Distributed:
          return <DistributedView run={run} worker={worker} span={span} />
        case Views.Memory:
          return <MemoryView run={run} worker={worker} span={span} />
        case Views.Module:
        case Views.Lightning:
          return <ModuleView run={run} worker={worker} span={span} />
      }
    } else {
      return (
        <DiffOverview
          run={diffLeftRun}
          worker={diffLeftWorker}
          span={diffLeftSpan}
          expRun={diffRightRun}
          expWorker={diffRightWorker}
          expSpan={diffRightSpan}
        />
      )
    }
  }

  const spanComponent = () => {
    const spanFragment = (
      <React.Fragment>
        <ListSubheader>Spans</ListSubheader>
        <ClickAwayListener onClickAway={SetIframeActive}>
          <FormControl variant="outlined" className={classes.formControl}>
            <Select value={span} onChange={handleSpanChange}>
              {spans.map((span) => (
                <MenuItem value={span}>{span}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </ClickAwayListener>
      </React.Fragment>
    )

    if (!spans || spans.length <= 1) {
      return <div className={classes.hide}>{spanFragment}</div>
    } else {
      return spanFragment
    }
  }

  return (
    <div className={classes.root}>
      <CssBaseline />
      <Drawer
        variant="permanent"
        anchor="left"
        className={clsx(classes.drawer, {
          [classes.drawerOpen]: open,
          [classes.drawerClose]: !open
        })}
        classes={{
          paper: clsx({
            [classes.drawerOpen]: open,
            [classes.drawerClose]: !open
          })
        }}
        onClick={SetIframeActive}
      >
        <div className={classes.toolbar}>
          <IconButton
            className={classes.iconButton}
            onClick={handleDrawerClose}
          >
            <ChevronLeftIcon />
          </IconButton>
        </div>
        <Divider />
        <Box>
          <Tabs
            value={selectedTab}
            onChange={handleTabChange}
            aria-label="basic tabs example"
          >
            <Tab label="Normal" />
            <Tab label="Diff" />
          </Tabs>
        </Box>
        {selectedTab == 0 ? (
          <>
            <ListSubheader>Runs</ListSubheader>
            <ClickAwayListener onClickAway={SetIframeActive}>
              <FormControl variant="outlined" className={classes.formControl}>
                <Select value={run} onChange={handleRunChange}>
                  {runs.map((run) => (
                    <MenuItem value={run}>{run}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </ClickAwayListener>
            <ListSubheader>Views</ListSubheader>
            <ClickAwayListener onClickAway={SetIframeActive}>
              <FormControl variant="outlined" className={classes.formControl}>
                <Select value={view} onChange={handleViewChange}>
                  {views.map((view) => (
                    <MenuItem value={view}>{ViewNames[view]}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </ClickAwayListener>
            <ListSubheader>Workers</ListSubheader>
            <ClickAwayListener onClickAway={SetIframeActive}>
              <FormControl variant="outlined" className={classes.formControl}>
                <Select value={worker} onChange={handleWorkerChange}>
                  {workers.map((worker) => (
                    <MenuItem value={worker}>{worker}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </ClickAwayListener>
            {spanComponent()}
          </>
        ) : (
          <>
            <Typography variant="h6">&nbsp;&nbsp;Baseline</Typography>
            <ListSubheader>Runs</ListSubheader>
            <FormControl variant="outlined" className={classes.formControl}>
              <Select value={diffLeftRun} onChange={handleDiffLeftRunChange}>
                {runs.map((run) => (
                  <MenuItem value={run}>{run}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <ListSubheader>Workers</ListSubheader>

            <FormControl variant="outlined" className={classes.formControl}>
              <Select
                value={diffLeftWorker}
                onChange={handleDiffLeftWorkerChange}
              >
                {diffLeftWorkerOptions.map((worker) => (
                  <MenuItem value={worker}>{worker}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <ListSubheader>Spans</ListSubheader>
            <FormControl variant="outlined" className={classes.formControl}>
              <Select value={diffLeftSpan} onChange={handleDiffLeftSpanChange}>
                {diffLeftSpansOptions.map((span) => (
                  <MenuItem value={span}>{span}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <Divider />

            <Typography variant="h6">&nbsp;&nbsp;Experimental</Typography>
            <ListSubheader>Runs</ListSubheader>
            <FormControl variant="outlined" className={classes.formControl}>
              <Select value={diffRightRun} onChange={handleDiffRightRunChange}>
                {runs.map((run) => (
                  <MenuItem value={run}>{run}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <ListSubheader>Workers</ListSubheader>
            <FormControl variant="outlined" className={classes.formControl}>
              <Select
                value={diffRightWorker}
                onChange={handleDiffRightWorkerChange}
              >
                {diffRightWorkerOptions.map((worker) => (
                  <MenuItem value={worker}>{worker}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <ListSubheader>Spans</ListSubheader>
            <FormControl variant="outlined" className={classes.formControl}>
              <Select
                value={diffRightSpan}
                onChange={handleDiffRightSpanChange}
              >
                {diffRightSpansOptions.map((span) => (
                  <MenuItem value={span}>{span}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </>
        )}
      </Drawer>
      {!open && (
        <Fab
          className={classes.fab}
          size="small"
          color="primary"
          aria-label="show menu"
          onClick={handleDrawerOpen}
        >
          <ChevronRightIcon />
        </Fab>
      )}
      <main className={classes.content}>{renderContent()}</main>
    </div>
  )
}
