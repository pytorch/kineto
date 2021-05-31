/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import Grid from '@material-ui/core/Grid'
import TextField, { TextFieldProps } from '@material-ui/core/TextField'
import CardHeader from '@material-ui/core/CardHeader'
import CardContent from '@material-ui/core/CardContent'
import { makeStyles } from '@material-ui/core/styles'
import MenuItem from '@material-ui/core/MenuItem'
import InputLabel from '@material-ui/core/InputLabel'
import Select, { SelectProps } from '@material-ui/core/Select'

import * as React from 'react'
import * as api from '../api'
import { MemoryData } from '../api'
import { DataLoading } from './DataLoading'
import { useSearchDirectly } from '../utils/search'
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
  view: string
}

export const MemoryView: React.FC<IProps> = (props) => {
  const { run, worker, view } = props
  const classes = useStyles()

  const [memoryData, setMemoryData] = React.useState<MemoryData | undefined>(
    undefined
  )
  const [devices, setDevices] = React.useState<string[]>([])
  const [device, setDevice] = React.useState('')
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
    api.defaultApi.memoryGet(run, worker).then((resp) => {
      setMemoryData(resp)
      setDevices(Object.keys(resp.data))
      setDevice(resp.metadata.default_device)
    })
  }, [run, worker, view])

  const onDeviceChanged: SelectProps['onChange'] = (event) => {
    setDevice(event.target.value as string)
  }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Memory View" />
        <CardContent>
          <Grid direction="column" container spacing={1}>
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
