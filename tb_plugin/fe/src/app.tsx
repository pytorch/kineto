/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

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
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'
import ChevronRightIcon from '@material-ui/icons/ChevronRight'
import Typography from '@material-ui/core/Typography'
import 'antd/es/button/style/css'
import 'antd/es/list/style/css'
import 'antd/es/table/style/css'
import clsx from 'clsx'
import * as React from 'react'
import * as api from './api'
import { DistributedView } from './components/DistributedView'
import { FullCircularProgress } from './components/FullCircularProgress'
import { Kernel } from './components/Kernel'
import { MemoryView } from './components/MemoryView'
import { Operator } from './components/Operator'
import { Overview } from './components/Overview'
import { TraceView } from './components/TraceView'
import { setup } from './setup'
import './styles.css'
import { firstOrUndefined, sleep } from './utils'

import Button from '@material-ui/core/Button'
import TextField from '@material-ui/core/TextField'
import Checkbox from '@material-ui/core/Checkbox'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Dialog from '@material-ui/core/Dialog'
import DialogActions from '@material-ui/core/DialogActions'
import DialogContent from '@material-ui/core/DialogContent'
import DialogContentText from '@material-ui/core/DialogContentText'
import DialogTitle from '@material-ui/core/DialogTitle'
import FormLabel from '@material-ui/core/FormLabel'
import Grid from '@material-ui/core/Grid'
import FormGroup from '@material-ui/core/FormGroup'
import Snackbar from '@material-ui/core/Snackbar'
import Alert, { Color } from '@material-ui/lab/Alert'
import Slide from '@material-ui/core/Slide'
import Backdrop from '@material-ui/core/Backdrop'
import CircularProgress from '@material-ui/core/CircularProgress'

export enum Views {
  Overview = 'Overview',
  Operator = 'Operator',
  Kernel = 'Kernel',
  Trace = 'Trace',
  Distributed = 'Distributed',
  Memory = 'Memory'
}

const ViewNames = {
  [Views.Overview]: Views.Overview,
  [Views.Operator]: Views.Operator,
  [Views.Kernel]: 'GPU Kernel',
  [Views.Trace]: Views.Trace,
  [Views.Distributed]: Views.Distributed,
  [Views.Memory]: Views.Memory
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
  },
  button: {
    padding: theme.spacing(2),
    margin: theme.spacing(1)
  },
  formLabel: {
    fontSize: '0.75rem',
    paddingTop: theme.spacing(2)
  },
  backdrop: {
    zIndex: theme.zIndex.drawer + 1,
    color: '#fff'
  }
}))

export const App = () => {
  const classes = useStyles()

  const [startDialogOpen, setStartDialogOpen] = React.useState(false)
  const [stopDialogOpen, setStopDialogOpen] = React.useState(false)
  const [isWaiting, setIsWaiting] = React.useState(false)
  const [profilingSettings, setProfilingSettings] = React.useState({
    host: 'localhost',
    port: 3180,
    log_dir: 'default',
    warmup_dur: 0,
    record_shapes: true,
    profile_memory: true,
    with_stack: true,
    with_flops: false
  })
  const [alertConfig, setAlertConfig] = React.useState({
    open: false,
    status: 'error',
    message: 'Undefined error message.'
  })

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

  const handleStartProfiling = () => {
    setStartDialogOpen(true)
  }

  const handleStopProfiling = () => {
    setStopDialogOpen(true)
  }

  const handleStartDialogSubmit = () => {
    setIsWaiting(true)
    setStartDialogOpen(false)
    api.defaultApi
      .servicePut('start', {
        ...profilingSettings,
        ['host']:
          profilingSettings.host.startsWith('http://') ||
          profilingSettings.host.startsWith('https://')
            ? profilingSettings.host
            : 'http://' + profilingSettings.host
      })
      .then((response) => {
        setAlertConfig({
          open: true,
          status: response.success ? 'success' : 'error',
          message: response.message
        })
      })
      .catch(() => {
        setAlertConfig({
          open: true,
          status: 'error',
          message: 'Failed to request profiling service to start.'
        })
      })
      .finally(() => {
        setIsWaiting(false)
      })
  }

  const handleStartDialogCancel = () => {
    setStartDialogOpen(false)
  }

  const dataSynchronize = async () => {
    try {
      if (run) {
        const rawViews = await api.defaultApi.viewsGet(run)
        const views = rawViews
          .map((v) => Views[Views[v as Views]])
          .filter(Boolean)
        setViews(views)
      }
    } catch (e) {
      console.info('Cannot fetch views: ', e)
    }
    try {
      if (run && view) {
        const workers = await api.defaultApi.workersGet(run, view)
        setWorkers(workers)
      }
    } catch (e) {
      console.info('Cannot fetch workers: ', e)
    }
    try {
      if (run && worker) {
        const spans = await api.defaultApi.spansGet(run, worker)
        setSpans(spans)
      }
    } catch (e) {
      console.info('Cannot fetch spans: ', e)
    }
  }

  const handleStopDialogSubmit = () => {
    setIsWaiting(true)
    setStopDialogOpen(false)
    api.defaultApi
      .servicePut('stop', {
        ...profilingSettings,
        ['host']:
          profilingSettings.host.startsWith('http://') ||
          profilingSettings.host.startsWith('https://')
            ? profilingSettings.host
            : 'http://' + profilingSettings.host
      })
      .then((response) => {
        if(response.success) {
          dataSynchronize().then(() => {
            setAlertConfig({
              open: true,
              status: 'success',
              message: response.message
            })
          })
        } else {
          setAlertConfig({
            open: true,
            status: 'error',
            message: response.message
          })
        }
      })
      .catch(() => {
        setAlertConfig({
          open: true,
          status: 'error',
          message: 'An error occurs in tb_plugin server.'
        })
      })
      .finally(() => {
        setIsWaiting(false)
      })
  }

  const handleStopDialogCancel = () => {
    setStopDialogOpen(false)
  }

  const handleCheckChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setProfilingSettings({
      ...profilingSettings,
      [event.target.name]: event.target.checked
    })
  }

  const handleNumberChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setProfilingSettings({
      ...profilingSettings,
      [event.target.name]: Number(event.target.value)
    })
  }

  const handleStringChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setProfilingSettings({
      ...profilingSettings,
      [event.target.name]: event.target.value
    })
  }

  const handleAlertClose = () => {
    setAlertConfig({
      ...alertConfig,
      ['open']: false
    })
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

  const [open, setOpen] = React.useState(true)

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
        <ListSubheader>Profiling</ListSubheader>
        <Button
          variant="outlined"
          className={classes.button}
          onClick={handleStartProfiling}
        >
          start
        </Button>
        <Button
          variant="outlined"
          className={classes.button}
          onClick={handleStopProfiling}
        >
          stop
        </Button>
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
      <Dialog
        open={startDialogOpen}
        onClose={handleStartDialogCancel}
        aria-labelledby="form-dialog-title"
      >
        <DialogTitle id="form-dialog-title">Start Profiling</DialogTitle>
        <DialogContent>
          <DialogContentText>
            To start profiling service, input the host and port of the PyTorch training process, run name, warmup duration and
            profiling configs here.
          </DialogContentText>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <TextField
                name="host"
                label="PyTorch Service Host"
                type="text"
                value={profilingSettings.host}
                onChange={handleStringChange}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                name="port"
                label="PyTorch Service Port"
                type="number"
                value={profilingSettings.port}
                onChange={handleNumberChange}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                name="log_dir"
                label="Run Name"
                type="text"
                value={profilingSettings.log_dir}
                onChange={handleStringChange}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                name="warmup_dur"
                label="Warmup Duration(Sec)"
                type="number"
                value={profilingSettings.warmup_dur}
                onChange={handleNumberChange}
              />
            </Grid>
          </Grid>
          <FormControl component="fieldset">
            <FormLabel className={classes.formLabel}>
              Profiling Config
            </FormLabel>
            <FormGroup aria-label="position" row>
              <Grid container>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={profilingSettings.record_shapes}
                        onChange={handleCheckChange}
                        name="record_shapes"
                        color="primary"
                      />
                    }
                    label="Record Shapes"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={profilingSettings.profile_memory}
                        onChange={handleCheckChange}
                        name="profile_memory"
                        color="primary"
                      />
                    }
                    label="Profile Memory"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={profilingSettings.with_stack}
                        onChange={handleCheckChange}
                        name="with_stack"
                        color="primary"
                      />
                    }
                    label="With Stack"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={profilingSettings.with_flops}
                        onChange={handleCheckChange}
                        name="with_flops"
                        color="primary"
                      />
                    }
                    label="With Flops"
                  />
                </Grid>
              </Grid>
            </FormGroup>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleStartDialogCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleStartDialogSubmit} color="primary">
            Submit
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={stopDialogOpen}
        onClose={handleStopDialogCancel}
        aria-labelledby="form-dialog-title"
      >
        <DialogTitle id="form-dialog-title">Stop Profiling</DialogTitle>
        <DialogContent>
          <DialogContentText>
            To stop profiling service, you need to specify the port and host of the PyTorch training process to
            send the stop message.
          </DialogContentText>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <TextField
                name="host"
                label="PyTorch Service Host"
                type="text"
                value={profilingSettings.host}
                onChange={handleStringChange}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                name="port"
                label="PyTorch Service Port"
                type="number"
                value={profilingSettings.port}
                onChange={handleNumberChange}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleStopDialogCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleStopDialogSubmit} color="primary">
            Submit
          </Button>
        </DialogActions>
      </Dialog>
      <Backdrop className={classes.backdrop} open={isWaiting}>
        <CircularProgress color="inherit" />
      </Backdrop>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={alertConfig.open}
        onClose={handleAlertClose}
        autoHideDuration={5000}
        TransitionComponent={Slide}
      >
        <Alert
          severity={alertConfig.status as Color}
          elevation={6}
          variant="filled"
        >
          {alertConfig.message}
        </Alert>
      </Snackbar>
      <main className={classes.content}>{renderContent()}</main>
    </div>
  )
}
