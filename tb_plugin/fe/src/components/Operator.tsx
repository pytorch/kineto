/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Grid from '@material-ui/core/Grid'
import GridList from '@material-ui/core/GridList'
import GridListTile from '@material-ui/core/GridListTile'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import Radio from '@material-ui/core/Radio'
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup'
import Select, { SelectProps } from '@material-ui/core/Select'
import { makeStyles } from '@material-ui/core/styles'
import TextField, {
  StandardTextFieldProps,
  TextFieldProps
} from '@material-ui/core/TextField'
import * as React from 'react'
import * as api from '../api'
import {
  OperationTableData,
  OperationTableDataInner,
  OperatorGraph
} from '../api'
import { OperationGroupBy } from '../constants/groupBy'
import { useSearchDirectly } from '../utils/search'
import { topIsValid, UseTop, useTopN } from '../utils/top'
import { PieChart } from './charts/PieChart'
import { DataLoading } from './DataLoading'
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers'
import { OperationTable } from './tables/OperationTable'
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
  inputWidthOverflow: {
    minWidth: '15em',
    whiteSpace: 'nowrap'
  },
  full: {
    width: '100%'
  },
  description: {
    marginLeft: theme.spacing(1)
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

export const Operator: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()
  const tooltipCommonClasses = useTooltipCommonStyles()
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses),
    [tooltipCommonClasses]
  )

  const [operatorGraph, setOperatorGraph] = React.useState<
    OperatorGraph | undefined
  >(undefined)
  const [operatorTable, setOperatorTable] = React.useState<
    OperationTableData | undefined
  >(undefined)
  const [sortColumn, setSortColumn] = React.useState('')
  const [tableTooltips, setTableTooltips] = React.useState<any | undefined>(
    undefined
  )
  const [groupBy, setGroupBy] = React.useState(OperationGroupBy.Operation)
  const [searchOperatorName, setSearchOperatorName] = React.useState('')
  const [topText, actualTop, useTop, setTopText, setUseTop] = useTopN({
    defaultUseTop: UseTop.Use,
    defaultTop: 10
  })

  const getName = React.useCallback(
    (row: OperationTableDataInner) => row.name,
    []
  )
  const [searchedOperatorTable] = useSearchDirectly(
    searchOperatorName,
    getName,
    operatorTable
  )

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string)
  }

  React.useEffect(() => {
    if (operatorGraph) {
      const counts = [
        operatorGraph.device_self_time?.rows.length ?? 0,
        operatorGraph.device_total_time?.rows.length ?? 0,
        operatorGraph.host_self_time.rows?.length ?? 0,
        operatorGraph.host_total_time.rows?.length ?? 0
      ]
      setTopText(String(Math.min(Math.max(...counts), 10)))
    }
  }, [operatorGraph])

  React.useEffect(() => {
    api.defaultApi
      .operationTableGet(run, worker, span, groupBy)
      .then((resp) => {
        setSortColumn(resp.metadata.sort)
        setTableTooltips(resp.metadata.tooltips)
        setOperatorTable(resp.data)
      })
  }, [run, worker, span, groupBy])

  React.useEffect(() => {
    api.defaultApi
      .operationGet(run, worker, span, OperationGroupBy.Operation)
      .then((resp) => {
        setOperatorGraph(resp)
      })
  }, [run, worker, span])

  const onGroupByChanged: SelectProps['onChange'] = (event) => {
    setGroupBy(event.target.value as OperationGroupBy)
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

  const renderCharts = (graph: api.OperatorGraph) => {
    return (
      <GridList className={classes.full} cellHeight="auto" cols={2}>
        {graph.device_self_time && (
          <GridListTile>
            <Card>
              {graph.device_self_time.title && (
                <CardHeader
                  title={chartHeaderRenderer(
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
                  title={chartHeaderRenderer(
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
                title={chartHeaderRenderer(
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
                title={chartHeaderRenderer(
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
                    value={topText}
                    onChange={onTopChanged}
                    error={!topIsValid(topText)}
                  />
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
                      <MenuItem value={OperationGroupBy.OperationAndInputShape}>
                        Operator + Input Shape
                      </MenuItem>
                      <MenuItem value={OperationGroupBy.Operation}>
                        Operator
                      </MenuItem>
                    </Select>
                  </Grid>
                  <Grid item>
                    <TextField
                      classes={{ root: classes.inputWidthOverflow }}
                      value={searchOperatorName}
                      onChange={onSearchOperatorChanged}
                      type="search"
                      label="Search by Name"
                    />
                  </Grid>
                </Grid>
              </Grid>
              <Grid>
                <DataLoading value={searchedOperatorTable}>
                  {(table) => (
                    <OperationTable
                      data={table}
                      groupBy={groupBy}
                      run={run}
                      span={span}
                      worker={worker}
                      sortColumn={sortColumn}
                      tooltips={tableTooltips}
                    />
                  )}
                </DataLoading>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  )
}
