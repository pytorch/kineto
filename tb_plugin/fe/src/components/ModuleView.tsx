/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/
import Card from '@material-ui/core/Card'
import CardHeader from '@material-ui/core/CardHeader'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import Select, { SelectProps } from '@material-ui/core/Select'
import { makeStyles } from '@material-ui/core/styles'
import { Table } from 'antd'
import * as React from 'react'
// @ts-ignore
import { FlameGraph } from 'react-flame-graph'
import { defaultApi, KeyedColumn, ModuleStats, ModuleViewData } from '../api'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  hide: {
    display: 'none'
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

const getKeyedTableColumns = function (columns: KeyedColumn[]): any {
  return columns.map(function (col: KeyedColumn) {
    return {
      dataIndex: col.key,
      key: col.key,
      title: col.name
    }
  })
}

const getTableRows = function (key: number, rows: ModuleStats[]): any {
  return rows.map(function (row: ModuleStats) {
    const data = {
      key: key++,
      name: row.name,
      occurences: row.occurences,
      operators: row.operators,
      host_duration: row.host_duration,
      self_host_duration: row.self_host_duration,
      device_duration: row.device_duration,
      self_device_duration: row.self_device_duration,
      children: getTableRows(key, row.children)
    }
    if (data.children.length == 0) {
      delete data.children
    }

    return data
  })
}

const getFlameGraphData = function (rows: ModuleStats[]): any {
  return rows.map(function (row: ModuleStats) {
    const data = {
      name: row.name,
      value: row.avg_duration,
      tooltip: `${row.name}: ${row.avg_duration} us`,
      children: getFlameGraphData(row.children)
    }

    if (data.children.length == 0) {
      delete data.children
    }

    return data
  })
}

export const ModuleView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [moduleView, setModuleView] = React.useState<
    ModuleViewData | undefined
  >(undefined)
  const [flameData, setFlameData] = React.useState([])
  const [modules, setModules] = React.useState<number[]>([])
  const [module, setModule] = React.useState<number>(0)

  const [columns, setColumns] = React.useState([])
  const [rows, setRows] = React.useState([])

  React.useEffect(() => {
    defaultApi.moduleGet(run, worker, span).then((resp) => {
      setModuleView(resp)
      if (resp) {
        // set the flamegraph data
        const flameData = getFlameGraphData(resp.data)
        setFlameData(flameData)
        setModules(Array.from(Array(flameData.length).keys()))
        setModule(0)

        // set the tree table data
        setColumns(getKeyedTableColumns(resp.columns))
        setRows(getTableRows(1, resp.data))
      }
    })
  }, [run, worker, span])

  const handleModuleChange: SelectProps['onChange'] = (event) => {
    setModule(event.target.value as number)
  }

  const moduleComponent = () => {
    const moduleFragment = (
      <React.Fragment>
        <InputLabel id="module-graph">Module</InputLabel>
        <Select value={module} onChange={handleModuleChange}>
          {modules.map((m) => (
            <MenuItem value={m}>{m}</MenuItem>
          ))}
        </Select>
      </React.Fragment>
    )

    if (!modules || modules.length <= 2) {
      return <div className={classes.hide}>{moduleFragment}</div>
    } else {
      return moduleFragment
    }
  }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Module View" />

        {/* defaultExpandAllRows will only valid when first render the Table
          if row is null, then it will be ignored so all data will be collapse.
          see https://segmentfault.com/a/1190000007830998 for more information.
          */}
        {rows && rows.length > 0 && (
          <Table
            size="small"
            bordered
            columns={columns}
            dataSource={rows}
            expandable={{
              defaultExpandAllRows: true
            }}
          />
        )}

        {moduleComponent()}

        {flameData && flameData.length > 0 && (
          <FlameGraph
            data={flameData[module]}
            height={200}
            width={800}
            onChange={(node: any) => {
              console.log(`"${node.name}" focused`)
            }}
          />
        )}
      </Card>
    </div>
  )
}
