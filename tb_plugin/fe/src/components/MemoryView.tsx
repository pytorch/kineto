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

export const MemoryView: React.FC<IProps> = React.memo((props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [memoryData, setMemoryData] = React.useState<MemoryData | undefined>(
    undefined
  )
  const [hasMemoryEventsData, setHasMemoryEventsData] = React.useState<
    boolean | undefined
  >(undefined)
  const [memoryEventsData, setMemoryEventsData] = React.useState<
    MemoryEventsData | undefined
  >(undefined)

  const [hasMemoryCurveGraph, setHasMemoryCurveGraph] = React.useState<
    boolean | undefined
  >(undefined)
  const [memoryCurveGraph, setMemoryCurveGraph] = React.useState<
    MemoryCurve | undefined
  >(undefined)
  const [devices, setDevices] = React.useState<string[]>([])
  const [device, setDevice] = React.useState('')
  interface SelectedRange {
    start: number
    end: number
    startTs: number
    endTs: number
  }
  const [selectedRange, setSelectedRange] = React.useState<
    SelectedRange | undefined
  >()
  const [searchOperatorName, setSearchOperatorName] = React.useState('')
  const [searchEventOperatorName, setSearchEventOperatorName] = React.useState(
    ''
  )

  const getSearchIndex = function () {
    if (!memoryData) {
      return -1
    }
    for (let i = 0; i < memoryData.columns.length; i++) {
      if (memoryData.columns[i].name == memoryData.metadata.search) {
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
    memoryData?.rows[device] ?? []
  )
  const [searchedEventsTableDataRows] = useSearchDirectly(
    searchEventOperatorName,
    getName,
    memoryEventsData?.rows[device] ?? []
  )

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string)
  }

  const onSearchEventOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchEventOperatorName(event.target.value as string)
  }

  React.useEffect(() => {
    api.defaultApi
      .memoryGet(
        run,
        worker,
        span,
        selectedRange?.startTs,
        selectedRange?.endTs
      )
      .then((resp) => {
        setMemoryData(resp)
        if (!devices || devices.length == 0) {
          // setDevices only execute on view load. Since selection on curve
          // might filter all events later, some devices might is missing.
          setDevices(Object.keys(resp.rows))
        }
      })
  }, [run, worker, span, selectedRange])

  React.useEffect(() => {
    api.defaultApi
      .memoryEventsGet(
        run,
        worker,
        span,
        selectedRange?.startTs,
        selectedRange?.endTs
      )
      .then((resp) => {
        if (hasMemoryEventsData === undefined) {
          setHasMemoryEventsData(Object.keys(resp.rows).length != 0)
        }
        setMemoryEventsData(resp)
      })
  }, [run, worker, span, selectedRange])

  React.useEffect(() => {
    api.defaultApi.memoryCurveGet(run, worker, span).then((resp) => {
      setDevice(resp.metadata.default_device)
      if (hasMemoryCurveGraph === undefined) {
        setHasMemoryCurveGraph(Object.keys(resp.rows).length != 0)
      }
      setMemoryCurveGraph(resp)
    })
  }, [run, worker, span])

  const onDeviceChanged: SelectProps['onChange'] = (event) => {
    setDevice(event.target.value as string)
    setSelectedRange(undefined)
  }

  const onSelectedRangeChanged = (start: number, end: number) => {
    let bias = memoryCurveGraph?.metadata.first_ts ?? 0
    let scale = 1 / (memoryCurveGraph?.metadata.time_factor ?? 1)
    let startTs = Math.round(start * scale + bias)
    let endTs = Math.round(end * scale + bias)
    if (startTs == endTs) {
      setSelectedRange(undefined)
      return
    }
    setSelectedRange({ start, end, startTs, endTs })
  }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Memory View" />
        <CardContent>
          <Grid direction="column" container spacing={1}>
            <Grid item>
              <DataLoading value={memoryCurveGraph}>
                {(graph) => (
                  <Grid container direction="column">
                    <Grid item>
                      <InputLabel id="memory-curve-device">Device</InputLabel>
                      <Select
                        labelId="memory-curve-device"
                        value={device}
                        onChange={onDeviceChanged}
                      >
                        {devices.map((device) => (
                          <MenuItem value={device}>{device}</MenuItem>
                        ))}
                      </Select>
                    </Grid>
                    {hasMemoryCurveGraph && (
                      <Grid item>
                        <div>
                          <LineChart
                            hAxisTitle={`Time (${graph.metadata.time_metric})`}
                            vAxisTitle={`Memory Usage (${graph.metadata.memory_metric})`}
                            graph={{
                              title: graph.metadata.peaks[device],
                              columns: graph.columns,
                              rows: graph.rows[device] ?? []
                            }}
                            initialSelectionStart={selectedRange?.start}
                            initialSelectionEnd={selectedRange?.end}
                            onSelectionChanged={onSelectedRangeChanged}
                          />
                        </div>
                      </Grid>
                    )}
                  </Grid>
                )}
              </DataLoading>
            </Grid>
            {hasMemoryEventsData && (
              <>
                <Grid item container direction="column" sm={6}>
                  <Grid item container direction="column" alignContent="center">
                    <TextField
                      classes={{ root: classes.inputWidthOverflow }}
                      value={searchEventOperatorName}
                      onChange={onSearchEventOperatorChanged}
                      type="search"
                      label="Search by Name"
                    />
                  </Grid>
                </Grid>
                <Grid item direction="column">
                  <DataLoading value={memoryEventsData}>
                    {(data) => {
                      return (
                        <AntTableChart
                          graph={{
                            columns: data.columns,
                            rows: searchedEventsTableDataRows ?? []
                          }}
                          initialPageSize={10}
                        />
                      )
                    }}
                  </DataLoading>
                </Grid>
              </>
            )}
            <Grid item container direction="column" sm={6}>
              <Grid item container direction="column" alignContent="center">
                <TextField
                  classes={{ root: classes.inputWidthOverflow }}
                  value={searchOperatorName}
                  onChange={onSearchOperatorChanged}
                  type="search"
                  label="Search by Name"
                />
              </Grid>
              <Grid>
                <DataLoading value={memoryData}>
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
})
