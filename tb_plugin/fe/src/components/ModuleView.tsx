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
import {
  defaultApi,
  KeyedColumn,
  ModuleStats,
  ModuleViewData,
  OperatorNode
} from '../api'

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

const getKeyedTableColumns = (columns: KeyedColumn[]): any => {
  return columns.map((col: KeyedColumn) => {
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
      tooltip: `${row.name} (module id: ${row.id}): ${row.avg_duration} us`,
      children: getFlameGraphData(row.children)
    }

    if (data.children.length == 0) {
      delete data.children
    }

    return data
  })
}

const getTreeHeight = (row: ModuleStats): number => {
  if (row.children && row.children.length > 0) {
    let max_child_height = 0
    for (const child of row.children) {
      const child_height = getTreeHeight(child)
      if (max_child_height < child_height) {
        max_child_height = child_height
      }
    }

    return max_child_height + 1
  } else {
    return 1
  }
}

const getOperatorTree = (level: number, row: OperatorNode, result: any[]) => {
  const data = {
    level: level,
    name: row.name,
    // type: row.type,
    start: row.start_time,
    end: row.end_time
  }
  result.push(data)
  if (row.children.length > 0) {
    for (const child of row.children) {
      getOperatorTree(level + 1, child, result)
    }
  }
}

export const ModuleView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [moduleView, setModuleView] = React.useState<
    ModuleViewData | undefined
  >(undefined)
  const [flameData, setFlameData] = React.useState([])
  const [flameHeight, setFlameHeight] = React.useState<number>(0)
  const [modules, setModules] = React.useState<number[]>([])
  const [module, setModule] = React.useState<number>(0)

  const [columns, setColumns] = React.useState([])
  const [rows, setRows] = React.useState([])

  const timelineRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    defaultApi
      .moduleGet(run, worker, span)
      .then((resp) => {
        setModuleView(resp)
        if (resp) {
          // set the flamegraph data
          const flameData = getFlameGraphData(resp.data)
          setFlameData(flameData)
          let flameHeight = 0
          for (const flame of flameData) {
            const height = getTreeHeight(flame)
            if (flameHeight < height) {
              flameHeight = height
            }
          }
          setFlameHeight(flameHeight * 25)
          setModules(Array.from(Array(flameData.length).keys()))
          setModule(0)

          // set the tree table data
          setColumns(getKeyedTableColumns(resp.columns))
          setRows(getTableRows(1, resp.data))
        }
      })
      .catch((e) => {
        // console.error(e.statusText)
        if (e.status == 404) {
          setModules([])
          setFlameData([])
          setRows([])
        }
      })

    if (timelineRef.current) {
      defaultApi.treeGet(run, worker, span).then((resp) => {
        if (resp) {
          const data = new google.visualization.DataTable()
          data.addColumn({ type: 'string', id: 'Layer' })
          data.addColumn({ type: 'string', id: 'Name' })
          data.addColumn({ type: 'string', role: 'tooltip' })
          data.addColumn({ type: 'number', id: 'Start' })
          data.addColumn({ type: 'number', id: 'End' })

          let timeline_data: any[] = []
          let max_level = 0
          getOperatorTree(0, resp, timeline_data)
          timeline_data.forEach((d) => {
            if (max_level < d.level) {
              max_level = d.level
            }
            data.addRow([
              d.level.toString(),
              d.name,
              `${d.name} Duration: ${d.end - d.start} us`,
              d.start / 1000.0, // the time unit is us returned from server, but the google charts only accept milliseconds here
              d.end / 1000.0
            ])
          })

          const chart = new google.visualization.Timeline(timelineRef.current)

          // console.info(timeline_data)
          const options = {
            height: (max_level + 1) * 50,
            tooltip: {
              isHtml: true
            }
          }
          chart.draw(data, options)
        }
      })
    }
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

    if (!modules || modules.length <= 1) {
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
            height={flameHeight}
            width={800}
            onChange={(node: any) => {
              console.log(`"${node.name}" focused`)
            }}
          />
        )}

        <div ref={timelineRef}></div>
      </Card>
    </div>
  )
}
