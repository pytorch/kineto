/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import Select, { SelectProps } from '@material-ui/core/Select'
import { makeStyles } from '@material-ui/core/styles'
import TextField, { TextFieldProps } from '@material-ui/core/TextField'
import * as React from 'react'
import * as api from '../api'
import { MemoryCurve, MemoryData, MemoryEventsData } from '../api'
import { useSearchDirectly } from '../utils/search'
import { LineChart } from './charts/LineChart'
import { AntTableChart } from './charts/AntTableChart'
import { DataLoading } from './DataLoading'
import { MemoryTable } from './tables/MemoryTable'
import { SelectionRange } from './SelectionRange'

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

export const MemoryView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [memoryData, setMemoryData] = React.useState<MemoryData | undefined>(
    undefined
  )
  const [memoryEventsData, setMemoryEventsData] = React.useState<
    MemoryEventsData | undefined
  >(undefined)
  const [memoryCurveGraph, setMemoryCurveGraph] = React.useState<
    MemoryCurve | undefined
  >(undefined)
  const [devices, setDevices] = React.useState<string[]>([])
  const [device, setDevice] = React.useState('')
  const [curveDevice, setCurveDevice] = React.useState('')
  const [selectedRange, setSelectedRange] = React.useState({start: 1426839090790973, end: 1726839091220818})
  const [searchOperatorName, setSearchOperatorName] = React.useState('')

  const tableData = memoryData ? memoryData.data[device] : undefined

  const getSearchIndex = function () {
    if (!tableData || !memoryData) {
      return -1
    }
    for (let i = 0; i < tableData.columns.length; i++) {
      if (tableData.columns[i].name == memoryData.metadata.search) {
        return i
      }
    }
    return -1
  }

  const searchIndex = getSearchIndex()
  const getName = React.useCallback((row: any) => row[searchIndex], [
    searchIndex
  ])
  const [searchedTableDataRows] = useSearchDirectly(
    searchOperatorName,
    getName,
    tableData?.rows
  )

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string)
  }

  React.useEffect(() => {
    api.defaultApi.memoryGet(run, worker, span, selectedRange.start, selectedRange.end).then((resp) => {
      setMemoryData(resp)
      setDevices(Object.keys(resp.data))
      setDevice(resp.metadata.default_device)
    })
  }, [run, worker, span, selectedRange])

  React.useEffect(() => {
    api.defaultApi.memoryEventsGet(run, worker, span, selectedRange.start, selectedRange.end).then((resp) => {
      setMemoryEventsData(resp)
    })
  }, [run, worker, span, selectedRange])

  React.useEffect(() => {
    api.defaultApi.memoryCurveGet(run, worker, span).then((resp) => {
      setCurveDevice(resp.metadata.default_device)
      setMemoryCurveGraph(resp)
    })
  }, [run, worker, span])

  const onDeviceChanged: SelectProps['onChange'] = (event) => {
    setDevice(event.target.value as string)
  }

  const onCurveDeviceChanged: SelectProps['onChange'] = (event) => {
    setCurveDevice(event.target.value as string)
  }

  const onSelectedRangeChanged = (start: number, end: number) => {
    setSelectedRange({start: Math.round(start * 1e9), end: Math.round(end * 1e9)})
  }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Memory View" />
        <CardContent>
          <Grid direction="column" container spacing={1}>
            <Grid item >
              <DataLoading value={memoryCurveGraph}>
                {(graph) => (
                  <Grid container direction="column">
                    <Grid item>
                      <InputLabel id="memory-curve-device">Device</InputLabel>
                      <Select
                        labelId="memory-curve-device"
                        value={curveDevice}
                        onChange={onCurveDeviceChanged}
                      >
                        {graph.metadata.devices.map((device) => (
                          <MenuItem value={device}>{device}</MenuItem>
                        ))}
                      </Select>
                    </Grid>
                    <Grid item>
                      <div>
                        <LineChart
                          hAxisTitle="Time (ms)"
                          vAxisTitle="Memory Usage (GB)"
                          graph={{
                            title: graph.metadata.peaks[curveDevice],
                            columns: graph.columns,
                            rows: graph.rows[curveDevice]
                          }}
                          initialSelectionStart={selectedRange.start}
                          initialSelectionEnd={selectedRange.end}
                          onSelectionChanged={onSelectedRangeChanged}
                        />
                      </div>
                    </Grid>
                  </Grid>
                )}
              </DataLoading>
            </Grid>
            <Grid item>
              <DataLoading value={memoryEventsData}>
                {(data) => <AntTableChart graph={data} initialPageSize={10} />}
              </DataLoading>
            </Grid>
            <Grid item container direction="column" spacing={1}>
              <Grid item>
                <Grid container justify="space-around">
                  <Grid item>
                    <InputLabel id="memory-device">Device</InputLabel>
                    <Select
                      labelId="memory-device"
                      value={device}
                      onChange={onDeviceChanged}
                    >
                      {devices.map((device) => (
                        <MenuItem value={device}>{device}</MenuItem>
                      ))}
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
                <DataLoading value={tableData}>
                  {(data) => (
                    <MemoryTable
                      data={{
                        rows: searchedTableDataRows,
                        columns: data.columns
                      }}
                      sort={memoryData!.metadata.sort}
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
