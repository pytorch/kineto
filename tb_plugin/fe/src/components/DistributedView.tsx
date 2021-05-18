/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import Grid from '@material-ui/core/Grid'
import TextField, {
  TextFieldProps,
  StandardTextFieldProps
} from '@material-ui/core/TextField'
import CardHeader from '@material-ui/core/CardHeader'
import CardContent from '@material-ui/core/CardContent'
import { makeStyles } from '@material-ui/core/styles'
import MenuItem from '@material-ui/core/MenuItem'
import InputLabel from '@material-ui/core/InputLabel'
import Select, { SelectProps } from '@material-ui/core/Select'
import * as React from 'react'
import { PieChart } from './charts/PieChart'
import { TableChart } from './charts/TableChart'
import * as api from '../api'
import { Graph } from '../api'
import { DistributedGraph } from '../api'
import { firstOrUndefined } from '../utils'
import { DataLoading } from './DataLoading'
import { topIsValid, UseTop, useTopN } from '../utils/top'
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup'
import Radio from '@material-ui/core/Radio'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import { useSearch } from '../utils/search'
import { useTooltipCommonStyles, makeChartHeaderRenderer } from './helpers'
import { GPUKernelTotalTimeTooltip } from './TooltipDescriptions'
import { KernelGroupBy } from '../constants/groupBy'
import { ColumnChart, ColumnChartData } from './charts/ColumnChart'

export interface IProps {
  run: string
  worker: string
  view: string
}

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  verticalInput: {
    display: 'flex',
    alignItems: 'center'
  },
  inputWidth: {
    width: '4em'
  },
  inputWidthOverflow: {
    minWidth: '15em',
    whiteSpace: 'nowrap'
  },
  description: {
    marginLeft: theme.spacing(1)
  }
}))

export const DistributedView: React.FC<IProps> = (props) => {
  let { run, worker, view } = props
  worker = 'worker1'
  const classes = useStyles()
  const tooltipCommonClasses = useTooltipCommonStyles()
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses),
    [tooltipCommonClasses]
  )

  const [kernelGraph, setKernelGraph] = React.useState<Graph | undefined>(
    undefined
  )
  const [kernelTable, setKernelTable] = React.useState<Graph | undefined>(
    undefined
  )
  const [overlapGraph, setOverlapGraph] = React.useState<
    DistributedGraph | undefined
  >(undefined)
  const [waittimeGraph, setWaittimeGraph] = React.useState<
    DistributedGraph | undefined
  >(undefined)
  const [commopsTableData, setCommopsTableData] = React.useState<
    any | undefined
  >(undefined)
  const [groupBy, setGroupBy] = React.useState(KernelGroupBy.Kernel)
  const [searchKernelName, setSearchKernelName] = React.useState('')
  const [commopsWorkers, setCommopsWorkers] = React.useState<string[]>([])
  const [overlapSteps, setOverlapSteps] = React.useState<string[]>([])
  const [waittimeSteps, setWaittimeSteps] = React.useState<string[]>([])
  const [overlapStep, setOverlapStep] = React.useState('')
  const [waittimeStep, setWaittimeStep] = React.useState('')
  const [commopsWorker, setCommopsWorker] = React.useState('')
  const [searchOpName, setSearchOpName] = React.useState('')
  const [sortColumn, setSortColumn] = React.useState(2)

  const [topText, actualTop, useTop, setTopText, setUseTop] = useTopN({
    defaultUseTop: UseTop.Use,
    defaultTop: 10
  })

  React.useEffect(() => {
    setSearchOpName('')
  }, [groupBy])

  React.useEffect(() => {
    setWaittimeStep(firstOrUndefined(waittimeSteps) ?? '')
  }, [waittimeSteps])

  React.useEffect(() => {
    setOverlapStep(firstOrUndefined(overlapSteps) ?? '')
  }, [overlapSteps])

  React.useEffect(() => {
    setCommopsWorker(firstOrUndefined(commopsWorkers) ?? '')
  }, [commopsWorkers])

  React.useEffect(() => {
    if (overlapGraph) {
      setTopText(overlapGraph.metadata.title)
    }
  }, [overlapGraph])

  React.useEffect(() => {
    api.defaultApi.kernelTableGet(run, worker, view, groupBy).then((resp) => {
      setKernelTable(resp.data)
    })
  }, [run, worker, view, groupBy])

  React.useEffect(() => {
    api.defaultApi
      .kernelGet(run, worker, view, KernelGroupBy.Kernel)
      .then((resp) => {
        setKernelGraph(resp.total)
      })
  }, [run, worker, view])

  React.useEffect(() => {
    api.defaultApi.distributedOverlapGet(run, 'All', view).then((resp) => {
      setOverlapGraph(resp)
      setOverlapSteps(Object.keys(resp.data))
    })
    api.defaultApi.distributedWaittimeGet(run, 'All', view).then((resp) => {
      setWaittimeGraph(resp)
      setWaittimeSteps(Object.keys(resp.data))
    })
    api.defaultApi.distributedCommopsGet(run, 'All', view).then((resp) => {
      setCommopsTableData(resp)
      setCommopsWorkers(Object.keys(resp))
    })
  }, [run, worker, view])

  const [searchedKernelTable] = useSearch(searchKernelName, 'name', kernelTable)
  const [searchedOpTable] = useSearch(
    searchOpName,
    'operator',
    searchedKernelTable
  )

  const onCommopsWorkerChanged: SelectProps['onChange'] = (event) => {
    setCommopsWorker(event.target.value as string)
  }

  const onSearchKernelChanged: TextFieldProps['onChange'] = (event) => {
    setSearchKernelName(event.target.value as string)
  }

  const onOverlapStepChanged: SelectProps['onChange'] = (event) => {
    setOverlapStep(event.target.value as string)
  }

  const onWaittimeStepChanged: SelectProps['onChange'] = (event) => {
    setWaittimeStep(event.target.value as string)
  }

  const onSearchOpChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOpName(event.target.value as string)
  }

  const onUseTopChanged: RadioGroupProps['onChange'] = (event) => {
    setUseTop(event.target.value as UseTop)
  }

  const onTopChanged = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTopText(event.target.value)
  }

  const inputProps: StandardTextFieldProps['inputProps'] = {
    min: 1
  }

  const GPUKernelTotalTimeTitle = React.useMemo(
    () => chartHeaderRenderer('Total Time (us)', GPUKernelTotalTimeTooltip),
    [chartHeaderRenderer]
  )

  const getColumnChartData = (
    distributedGraph?: DistributedGraph,
    step?: string
  ) => {
    if (!distributedGraph || !step) return undefined
    const barLabels = Object.keys(distributedGraph.data[step])
    return {
      legends: distributedGraph.metadata.legends,
      barLabels,
      barHeights: barLabels.map((label) => distributedGraph.data[step][label])
    }
  }
  const overlapData = getColumnChartData(overlapGraph, overlapStep)
  const waittimeData = getColumnChartData(waittimeGraph, waittimeStep)

  const getTableData = (tableData?: any, worker?: string) => {
    if (!tableData || !worker) return undefined
    return tableData[worker] as Graph
  }
  const commopsTable = getTableData(commopsTableData, commopsWorker)

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Distributed View" />
        <CardContent>
          <Grid container spacing={1}>
            <Grid item sm={6}>
              <DataLoading value={overlapData}>
                {(chartData) => (
                  <Card elevation={0}>
                    <CardContent>
                      <Grid item container spacing={1} alignItems="center">
                        <Grid item>
                          <InputLabel id="overlap-step">Step</InputLabel>
                        </Grid>
                        <Grid item>
                          <Select
                            labelId="overlap-step"
                            value={overlapStep}
                            onChange={onOverlapStepChanged}
                          >
                            {overlapSteps.map((step) => (
                              <MenuItem value={step}>{step}</MenuItem>
                            ))}
                          </Select>
                        </Grid>
                      </Grid>
                    </CardContent>
                    <ColumnChart
                      title={overlapGraph?.metadata?.title}
                      chartData={chartData}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>

            <Grid item sm={6}>
              <DataLoading value={waittimeData}>
                {(chartData) => (
                  <Card elevation={0}>
                    <CardContent>
                      <Grid item container spacing={1} alignItems="center">
                        <Grid item>
                          <InputLabel id="waittime-step">Step</InputLabel>
                        </Grid>
                        <Grid item>
                          <Select
                            labelId="waittime-step"
                            value={waittimeStep}
                            onChange={onWaittimeStepChanged}
                          >
                            {waittimeSteps.map((step) => (
                              <MenuItem value={step}>{step}</MenuItem>
                            ))}
                          </Select>
                        </Grid>
                      </Grid>
                    </CardContent>
                    <ColumnChart
                      title={waittimeGraph?.metadata?.title}
                      chartData={chartData}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item sm={12}>
              <Grid container direction="column" spacing={0}>
                <Card elevation={0}>
                  <CardContent>
                    <Grid item container spacing={1} alignItems="center">
                      <Grid item>
                        <InputLabel id="table-worker">Worker</InputLabel>
                      </Grid>
                      <Grid item>
                        <Select
                          labelId="table-worker"
                          value={commopsWorker}
                          onChange={onCommopsWorkerChanged}
                        >
                          {commopsWorkers.map((worker) => (
                            <MenuItem value={worker}>{worker}</MenuItem>
                          ))}
                        </Select>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
                <Grid item>
                  <DataLoading value={commopsTable}>
                    {(graph) => <TableChart graph={graph} />}
                  </DataLoading>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  )
}
