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
import { AntTableChart } from './charts/AntTableChart'
import * as api from '../api'
import { Graph } from '../api'
import { DataLoading } from './DataLoading'
import { topIsValid, UseTop, useTopN } from '../utils/top'
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup'
import Radio from '@material-ui/core/Radio'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import { useSearch } from '../utils/search'
import { useTooltipCommonStyles, makeChartHeaderRenderer } from './helpers'
import { GPUKernelTotalTimeTooltip } from './TooltipDescriptions'
import { KernelGroupBy } from '../constants/groupBy'

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

export const Kernel: React.FC<IProps> = (props) => {
  const { run, worker, view } = props
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
  const [groupBy, setGroupBy] = React.useState(KernelGroupBy.Kernel)
  const [searchKernelName, setSearchKernelName] = React.useState('')
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
    if (kernelGraph) {
      setTopText(String(Math.min(kernelGraph.rows?.length, 10)))
    }
  }, [kernelGraph])

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

  const [searchedKernelTable] = useSearch(searchKernelName, 'name', kernelTable)
  const [searchedOpTable] = useSearch(
    searchOpName,
    'operator',
    searchedKernelTable
  )

  const onGroupByChanged: SelectProps['onChange'] = (event) => {
    setGroupBy(event.target.value as KernelGroupBy)
    setSortColumn(event.target.value == KernelGroupBy.Kernel ? 2 : 3)
  }

  const onSearchKernelChanged: TextFieldProps['onChange'] = (event) => {
    setSearchKernelName(event.target.value as string)
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

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Kernel View" />
        <CardContent>
          <Grid container direction="column" spacing={1}>
            <Grid item container>
              <Grid item>
                <RadioGroup row value={useTop} onChange={onUseTopChanged}>
                  <FormControlLabel
                    value={UseTop.NotUse}
                    control={<Radio />}
                    label="All kernels"
                  />
                  <FormControlLabel
                    value={UseTop.Use}
                    control={<Radio />}
                    label="Top kernels to show"
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
            <Grid item sm={6}>
              <DataLoading value={kernelGraph}>
                {(graph) => (
                  <Card elevation={0}>
                    <CardHeader title={GPUKernelTotalTimeTitle} />
                    <PieChart
                      title={graph.title}
                      graph={graph}
                      top={actualTop}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item container direction="column" spacing={1} sm={12}>
              <Grid item container>
                <Grid sm={6} item container justify="space-around">
                  <Grid item>
                    <InputLabel id="kernel-group-by">Group By</InputLabel>
                    <Select
                      labelId="kernel-group-by"
                      value={groupBy}
                      onChange={onGroupByChanged}
                    >
                      <MenuItem value={KernelGroupBy.KernelNameAndOpName}>
                        Kernel Name + Op Name
                      </MenuItem>
                      <MenuItem value={KernelGroupBy.Kernel}>
                        Kernel Name
                      </MenuItem>
                    </Select>
                  </Grid>
                </Grid>
                <Grid sm={6} item container spacing={1}>
                  <Grid item>
                    <TextField
                      classes={{ root: classes.inputWidthOverflow }}
                      value={searchKernelName}
                      onChange={onSearchKernelChanged}
                      type="search"
                      label="Search by Kernel Name"
                    />
                  </Grid>
                  {groupBy === KernelGroupBy.KernelNameAndOpName && (
                    <Grid item>
                      <TextField
                        classes={{ root: classes.inputWidthOverflow }}
                        value={searchOpName}
                        onChange={onSearchOpChanged}
                        type="search"
                        label="Search by Operator Name"
                      />
                    </Grid>
                  )}
                </Grid>
              </Grid>
              <Grid item>
                <DataLoading value={searchedOpTable}>
                  {(graph) => (
                    <AntTableChart graph={graph} sortColumn={sortColumn} />
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
