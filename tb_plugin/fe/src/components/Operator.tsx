/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import Grid from '@material-ui/core/Grid'
import TextField, {
  StandardTextFieldProps,
  TextFieldProps
} from '@material-ui/core/TextField'
import CardHeader from '@material-ui/core/CardHeader'
import CardContent from '@material-ui/core/CardContent'
import { makeStyles } from '@material-ui/core/styles'
import MenuItem from '@material-ui/core/MenuItem'
import InputLabel from '@material-ui/core/InputLabel'
import GridList from '@material-ui/core/GridList'
import GridListTile from '@material-ui/core/GridListTile'
import Typography from '@material-ui/core/Typography'
import Select, { SelectProps } from '@material-ui/core/Select'
import Tooltip from '@material-ui/core/Tooltip'

import * as React from 'react'
import { PieChart } from './charts/PieChart'
import { TableChart } from './charts/TableChart'
import * as api from '../api'
import { Graph, OperatorGraph } from '../api'
import { DataLoading } from './DataLoading'
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup'
import Radio from '@material-ui/core/Radio'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import { UseTop, useTopN } from '../utils/top'
import { useSearch } from '../utils/search'
import HelpOutline from '@material-ui/icons/HelpOutline'
import {
  DeviceSelfTimeTooltip,
  DeviceTotalTimeTooltip,
  HostSelfTimeTooltip,
  HostTotalTimeTooltip
} from './TooltipDescriptions'

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
  full: {
    width: '100%'
  },
  description: {
    marginLeft: theme.spacing(1)
  },
  tooltip: {
    maxWidth: '500px',
    whiteSpace: 'pre-wrap'
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center',
    fontSize: '.8rem',
    fontWeight: 'bold'
  },
  titleText: {
    marginRight: '6px'
  }
}))

export interface IProps {
  run: string
  worker: string
  view: string
}

enum GroupBy {
  Operation = 'Operation',
  OperationAndInputShape = 'OperationAndInputShape'
}

export const Operator: React.FC<IProps> = (props) => {
  const { run, worker, view } = props
  const classes = useStyles()

  const [operatorGraph, setOperatorGraph] = React.useState<
    OperatorGraph | undefined
  >(undefined)
  const [operatorTable, setOperatorTable] = React.useState<Graph | undefined>(
    undefined
  )
  const [groupBy, setGroupBy] = React.useState(GroupBy.Operation)
  const [searchOperatorName, setSearchOperatorName] = React.useState('')
  const [top, actualTop, useTop, setTop, setUseTop] = useTopN({
    defaultUseTop: UseTop.Use,
    defaultTop: 10
  })

  React.useEffect(() => {
    if (operatorGraph) {
      const counts = [
        operatorGraph.device_self_time?.rows.length ?? 0,
        operatorGraph.device_total_time?.rows.length ?? 0,
        operatorGraph.host_self_time.rows?.length ?? 0,
        operatorGraph.host_total_time.rows?.length ?? 0
      ]
      setTop(Math.min(Math.max(...counts), 10))
    }
  }, [operatorGraph])

  React.useEffect(() => {
    api.defaultApi
      .operationTableGet(run, worker, view, groupBy)
      .then((resp) => {
        setOperatorTable(resp.data)
      })
  }, [run, worker, view, groupBy])

  React.useEffect(() => {
    api.defaultApi
      .operationGet(run, worker, view, GroupBy.Operation)
      .then((resp) => {
        setOperatorGraph(resp)
      })
  }, [run, worker, view])

  const [searchedOperatorTable] = useSearch(
    searchOperatorName,
    'name',
    operatorTable
  )

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string)
  }

  const onGroupByChanged: SelectProps['onChange'] = (event) => {
    setGroupBy(event.target.value as GroupBy)
  }

  const onUseTopChanged: RadioGroupProps['onChange'] = (event) => {
    setUseTop(event.target.value as UseTop)
  }

  const onTopChanged = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTop(Number(event.target.value))
  }

  const inputProps: StandardTextFieldProps['inputProps'] = {
    min: 1
  }

  const makeChartHeader = (title: string, tooltip: string) => {
    return (
      <span className={classes.cardTitle}>
        <span className={classes.titleText}>{title}</span>
        <Tooltip arrow classes={{ tooltip: classes.tooltip }} title={tooltip}>
          <HelpOutline fontSize="small" />
        </Tooltip>
      </span>
    )
  }

  const renderCharts = (graph: api.OperatorGraph) => {
    return (
      <GridList className={classes.full} cellHeight="auto" cols={2}>
        {graph.device_self_time && (
          <GridListTile>
            <Card>
              {graph.device_self_time.title && (
                <CardHeader
                  title={makeChartHeader(
                    graph.device_self_time.title,
                    DeviceSelfTimeTooltip
                  )}
                />
              )}
              <PieChart graph={graph.device_self_time} top={actualTop} />
            </Card>
          </GridListTile>
        )}
        {graph.device_total_time && (
          <GridListTile>
            <Card>
              {graph.device_total_time.title && (
                <CardHeader
                  title={makeChartHeader(
                    graph.device_total_time.title,
                    DeviceTotalTimeTooltip
                  )}
                />
              )}
              <PieChart graph={graph.device_total_time} top={actualTop} />
            </Card>
          </GridListTile>
        )}
        <GridListTile>
          <Card>
            {graph.host_self_time.title && (
              <CardHeader
                title={makeChartHeader(
                  graph.host_self_time.title,
                  HostSelfTimeTooltip
                )}
              />
            )}
            <PieChart graph={graph.host_self_time} top={actualTop} />
          </Card>
        </GridListTile>
        <GridListTile>
          <Card>
            {graph.host_total_time.title && (
              <CardHeader
                title={makeChartHeader(
                  graph.host_total_time.title,
                  HostTotalTimeTooltip
                )}
              />
            )}
            <PieChart graph={graph.host_total_time} top={actualTop} />
          </Card>
        </GridListTile>
      </GridList>
    )
  }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Operator View" />
        <CardContent>
          <Grid direction="column" container spacing={1}>
            <Grid container item md={12}>
              <Grid item>
                <RadioGroup row value={useTop} onChange={onUseTopChanged}>
                  <FormControlLabel
                    value={UseTop.NotUse}
                    control={<Radio />}
                    label="All operators"
                  />
                  <FormControlLabel
                    value={UseTop.Use}
                    control={<Radio />}
                    label="Top operators to show"
                  />
                </RadioGroup>
              </Grid>
              {useTop === UseTop.Use && (
                <Grid item className={classes.verticalInput}>
                  <TextField
                    classes={{ root: classes.inputWidth }}
                    inputProps={inputProps}
                    type="number"
                    value={top}
                    onChange={onTopChanged}
                  />
                  <span className={classes.description}>(microseconds)</span>
                </Grid>
              )}
            </Grid>
            <Grid container item md={12}>
              <DataLoading value={operatorGraph}>{renderCharts}</DataLoading>
            </Grid>
            <Grid item container direction="column" spacing={1}>
              <Grid item>
                <Grid container justify="space-around">
                  <Grid item>
                    <InputLabel id="operator-group-by">Group By</InputLabel>
                    <Select
                      labelId="operator-group-by"
                      value={groupBy}
                      onChange={onGroupByChanged}
                    >
                      <MenuItem value={GroupBy.OperationAndInputShape}>
                        Operator + Input Shape
                      </MenuItem>
                      <MenuItem value={GroupBy.Operation}>Operator</MenuItem>
                    </Select>
                  </Grid>
                  <Grid item>
                    <TextField
                      value={searchOperatorName}
                      onChange={onSearchOperatorChanged}
                      type="search"
                      label="Search by Name"
                    />
                  </Grid>
                </Grid>
              </Grid>
              <Grid item>
                <DataLoading value={searchedOperatorTable}>
                  {(graph) => <TableChart graph={graph} />}
                </DataLoading>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  )
}
