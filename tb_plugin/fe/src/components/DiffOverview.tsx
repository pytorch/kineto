/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Button from '@material-ui/core/Button'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import Typography from '@material-ui/core/Typography'
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'
import { Select, Table } from 'antd'
import * as React from 'react'
import * as api from '../api'
import { useResizeEventDependency } from '../utils/resize'
import { FullCircularProgress } from './FullCircularProgress'

const { Option } = Select

const topGraphHeight = 230

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  pre: {
    '& ul': {
      margin: 0,
      paddingLeft: theme.spacing(3),
      ...theme.typography.body1
    },
    '& li': {},
    '& a': {
      color: '#ffa726'
    },
    '& a:active': {
      color: '#ffa726'
    },
    '& p': {
      margin: 0,
      ...theme.typography.subtitle1,
      fontWeight: theme.typography.fontWeightBold
    }
  },
  topGraph: {
    height: topGraphHeight + 40
  },
  iconButton: {
    padding: '8px'
  }
}))

export interface DiffColumnChartIProps {
  rawData: any[]
  selectCallback: (row: number, column: number) => void
}

export interface DiffStepChartIProps {
  rawData: any[]
}

const DiffColumnChart: React.FC<DiffColumnChartIProps> = (
  props: DiffColumnChartIProps
) => {
  const { rawData, selectCallback } = props
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    let left_duration_data: number[] = []
    let left_accumulated_duration_data: number[] = []

    let right_duration_data: number[] = []
    let right_accumulated_duration_data: number[] = []

    for (let i = 0; i < rawData.length; i++) {
      let curr = rawData[i]
      left_duration_data.push(curr[1])
      right_duration_data.push(curr[2])
      left_accumulated_duration_data.push(curr[3])
      right_accumulated_duration_data.push(curr[4])
    }

    let left_duration_max = Math.max(...left_duration_data)
    let right_duration_max = Math.max(...right_duration_data)
    let duration_max = Math.max(left_duration_max, right_duration_max)

    let left_accumulated_duration_max = Math.max(
      ...left_accumulated_duration_data
    )
    let right_accumulated_duration_max = Math.max(
      ...right_accumulated_duration_data
    )
    let accumulated_max = Math.max(
      left_accumulated_duration_max,
      right_accumulated_duration_max
    )

    var options = {
      title: 'Execution Comparsion',
      height: 500,
      seriesType: 'bars',
      series: {
        0: { type: 'bars', targetAxisIndex: 0 },
        1: { type: 'bars', targetAxisIndex: 0 },
        2: { type: 'line', targetAxisIndex: 1 },
        3: { type: 'line', targetAxisIndex: 1 }
      },
      vAxes: {
        0: {
          logScale: false,
          maxValue: duration_max
        },
        1: {
          logScale: false,
          maxValue: accumulated_max
        }
      }
    }

    const chart = new google.visualization.ComboChart(element)
    const data = google.visualization.arrayToDataTable(rawData)
    chart.draw(data, options)

    google.visualization.events.addListener(chart, 'select', (entry: any) => {
      var selectedItem = chart.getSelection()[0]
      if (selectedItem && selectedItem.hasOwnProperty('row')) {
        selectCallback(selectedItem.row, selectedItem.column)
      }
    })

    return () => {
      chart.clearChart()
    }
  }, [rawData, resizeEventDependency])

  return (
    <div>
      <div ref={graphRef}></div>
    </div>
  )
}

const DiffStepChart: React.FC<DiffStepChartIProps> = (
  props: DiffStepChartIProps
) => {
  const { rawData } = props
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    var options = {
      title: 'Execution Diff',
      height: 500
    }

    const chart = new google.visualization.SteppedAreaChart(element)
    const data = google.visualization.arrayToDataTable(rawData)
    chart.draw(data, options)

    return () => {
      chart.clearChart()
    }
  }, [rawData, resizeEventDependency])

  return (
    <div>
      <div ref={graphRef}></div>
    </div>
  )
}

export interface IProps {
  run: string
  worker: string
  span: string
  expRun: string
  expWorker: string
  expSpan: string
}

export interface ColumnUnderlyingData {
  name: string
  path: string
  leftAggs: any[]
  rightAggs: any[]
}

export interface TableRow {
  key: number

  operator: string
  baselineCalls?: number
  expCalls?: number
  deltaCalls?: number
  deltaCallsPercentNumber?: number
  deltaCallsPercent?: string

  baselineHostDuration: number
  expHostDuration: number
  deltaHostDuration: number
  deltaHostDurationPercentNumber: number
  deltaHostDurationPercent: string

  baselineSelfHostDuration: number
  expSelfHostDuration: number
  deltaSelfHostDuration: number
  deltaSelfHostDurationPercentNumber: number
  deltaSelfHostDurationPercent: string

  baselineDeviceDuration: number
  expDeviceDuration: number
  deltaDeviceDuration: number
  deltaDeviceDurationPercentNumber: number
  deltaDeviceDurationPercent: string

  baselineSelfDeviceDuration: number
  expSelfDeviceDuration: number
  deltaSelfDeviceDuration: number
  deltaSelfDeviceDurationPercentNumber: number
  deltaSelfDeviceDurationPercent: string
}

let columnChartDataStack: any[][] = []
let stepChartDataStack: any[][] = []
let columnUnderlyingDataStack: ColumnUnderlyingData[][] = []
let columnTableDataSourceStack: TableRow[][] = []

export const DiffOverview: React.FC<IProps> = (props: IProps) => {
  // #region - Constant

  const COMPOSITE_NODES_NAME = 'CompositeNodes'

  const hostDurationColumns = [
    {
      title: 'Baseline Host Duration (us)',
      dataIndex: 'baselineHostDuration',
      key: 'baselineHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineHostDuration - b.baselineHostDuration
    },
    {
      title: 'Exp Host Duration (us)',
      dataIndex: 'expHostDuration',
      key: 'expHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expHostDuration - b.expHostDuration
    },
    {
      title: 'Delta Host Duration (us)',
      dataIndex: 'deltaHostDuration',
      key: 'deltaHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaHostDuration! - b.deltaHostDuration!
    },
    {
      title: 'Delta Host Duration%',
      dataIndex: 'deltaHostDurationPercent',
      key: 'deltaHostDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaHostDurationPercentNumber! - b.deltaHostDurationPercentNumber!
    }
  ]

  const selfHostDurationColumns = [
    {
      title: 'Baseline Self Host Duration (us)',
      dataIndex: 'baselineSelfHostDuration',
      key: 'baselineSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineSelfHostDuration - b.baselineSelfHostDuration
    },
    {
      title: 'Exp Self Host Duration (us)',
      dataIndex: 'expSelfHostDuration',
      key: 'expSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expSelfHostDuration - b.expSelfHostDuration
    },
    {
      title: 'Delta Self Host Duration (us)',
      dataIndex: 'deltaSelfHostDuration',
      key: 'deltaSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfHostDuration! - b.deltaSelfHostDuration!
    },
    {
      title: 'Delta Self Host Duration%',
      dataIndex: 'deltaSelfHostDurationPercent',
      key: 'deltaSelfHostDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfHostDurationPercentNumber! -
        b.deltaSelfHostDurationPercentNumber!
    }
  ]

  const deviceDurationColumns = [
    {
      title: 'Baseline Device Duration (us)',
      dataIndex: 'baselineDeviceDuration',
      key: 'baselineDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineDeviceDuration - b.baselineDeviceDuration
    },
    {
      title: 'Exp Device Duration (us)',
      dataIndex: 'expDeviceDuration',
      key: 'expDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expDeviceDuration - b.expDeviceDuration
    },
    {
      title: 'Delta Device Duration (us)',
      dataIndex: 'deltaDeviceDuration',
      key: 'deltaDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaDeviceDuration! - b.deltaDeviceDuration!
    },
    {
      title: 'Delta Device Duration%',
      dataIndex: 'deltaDeviceDurationPercent',
      key: 'deltaDeviceDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaDeviceDurationPercentNumber! -
        b.deltaDeviceDurationPercentNumber!
    }
  ]

  const selfDeviceDurationColumns = [
    {
      title: 'Baseline Self Device Duration (us)',
      dataIndex: 'baselineSelfDeviceDuration',
      key: 'baselineSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineSelfDeviceDuration - b.baselineSelfDeviceDuration
    },
    {
      title: 'Exp Self Device Duration (us)',
      dataIndex: 'expSelfDeviceDuration',
      key: 'expSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expSelfDeviceDuration - b.expSelfDeviceDuration
    },
    {
      title: 'Delta Self Device Duration (us)',
      dataIndex: 'deltaSelfDeviceDuration',
      key: 'deltaSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfDeviceDuration! - b.deltaSelfDeviceDuration!
    },
    {
      title: 'Delta Self Device Duration%',
      dataIndex: 'deltaSelfDeviceDurationPercent',
      key: 'deltaSelfDeviceDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfDeviceDurationPercentNumber! -
        b.deltaSelfDeviceDurationPercentNumber!
    }
  ]

  type IColumnMapType = { [key: string]: any }

  const tableSourceColumnMap: IColumnMapType = {
    selfHostDuration: selfHostDurationColumns,
    hostDuration: hostDurationColumns,
    deviceDuration: deviceDurationColumns,
    selfDeviceDuration: selfDeviceDurationColumns
  }

  const baseTableColumns = [
    {
      title: 'Operator',
      dataIndex: 'operator',
      key: 'operator',
      sorter: (a: TableRow, b: TableRow) => a.operator.localeCompare(b.operator)
    },
    {
      title: 'Baseline Calls',
      dataIndex: 'baselineCalls',
      key: 'baselineCalls',
      sorter: (a: TableRow, b: TableRow) => a.baselineCalls! - b.baselineCalls!
    },
    {
      title: 'Exp Calls',
      dataIndex: 'expCalls',
      key: 'expCalls',
      sorter: (a: TableRow, b: TableRow) => a.expCalls! - b.expCalls!
    },
    {
      title: 'Delta Calls',
      dataIndex: 'deltaCalls',
      key: 'deltaCalls',
      sorter: (a: TableRow, b: TableRow) => a.deltaCalls! - b.deltaCalls!
    },
    {
      title: 'Delta Calls%',
      dataIndex: 'deltaCallsPercent',
      key: 'deltaCallsPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaCallsPercentNumber! - b.deltaCallsPercentNumber!
    }
  ]

  // #endregion

  // #region - State
  const [tableDataSource, setTableDataSource] = React.useState<TableRow[]>([])
  const { run, worker, span, expRun, expWorker, expSpan } = props

  const [columnUnderlyingData, setColumnUnderlyingData] = React.useState<
    ColumnUnderlyingData[]
  >([])

  const [
    rootUnderlyingData,
    setRootUnderlyingData
  ] = React.useState<ColumnUnderlyingData>()

  const [columnChartData, setColumnChartData] = React.useState<any[]>([])
  const [stepChartData, setStepChartData] = React.useState<any[]>([])

  const [
    selectedTableColumnsOptions,
    setSelectedTableColumnsOptions
  ] = React.useState<[key: string]>(['hostDuration'])
  const [selectedTableColumns, setSelectedTableColumns] = React.useState<any[]>(
    [...baseTableColumns, ...hostDurationColumns]
  )

  const [dataStackLevel, setDataStackLevel] = React.useState(0)
  const [loading, setLoading] = React.useState(false)

  // #endregion
  const classes = useStyles()

  // #region - Event Handler
  const handleChartColumnSelect = (row: number, column: number) => {
    if (columnUnderlyingData.length === 0) {
      return
    }

    let selectedUnderlyingData = columnUnderlyingData[row]
    if (!selectedUnderlyingData) {
      return
    }

    let tableDataSource = generateDataSourceFromUnderlyingData(
      selectedUnderlyingData
    )
    setTableDataSource(tableDataSource)
    columnTableDataSourceStack.push(tableDataSource)

    setLoading(true)

    api.defaultApi
      .diffnodeGet(
        run,
        worker,
        span,
        expRun,
        expWorker,
        expSpan,
        selectedUnderlyingData.path
      )
      .then((resp) => handleDiffNodeResp(resp))
      .finally(() => setLoading(false))
  }

  const handleGoBack = () => {
    if (columnChartDataStack.length > 1) {
      columnChartDataStack.pop()
      let top = columnChartDataStack[columnChartDataStack.length - 1]
      setColumnChartData(top)
    }

    if (stepChartDataStack.length > 1) {
      stepChartDataStack.pop()
      let top = stepChartDataStack[stepChartDataStack.length - 1]
      setStepChartData(top)
    }

    if (columnUnderlyingDataStack.length > 0) {
      columnUnderlyingDataStack.pop()
      let top = columnUnderlyingDataStack[columnUnderlyingDataStack.length - 1]
      setColumnUnderlyingData(top)
    }

    if (columnTableDataSourceStack.length > 0) {
      columnTableDataSourceStack.pop()
      let top =
        columnTableDataSourceStack[columnTableDataSourceStack.length - 1]

      if (top) {
        setTableDataSource(top)
      } else {
        let tableDataSource = generateDataSourceFromUnderlyingData(
          rootUnderlyingData!
        )
        setTableDataSource(tableDataSource)
      }
    }

    setDataStackLevel(dataStackLevel - 1)
  }

  const toPercentString = (percentNumber: number) => {
    if (isNaN(percentNumber)) {
      return 'N/A'
    }

    return `${percentNumber.toFixed(2)}%`
  }

  const handleColumnSelectionChange = (value: [key: string]) => {
    let columns = value.map((x) => tableSourceColumnMap[x]).flat()
    let r = [...baseTableColumns, ...columns]
    setSelectedTableColumnsOptions(value)
    setSelectedTableColumns(r)
  }

  const generateDataSourceFromUnderlyingData = (
    selectedUnderlyingData: ColumnUnderlyingData
  ) => {
    let tableDataSource: TableRow[] = []
    let tempLeftNameSet: string[] = []
    let tempRightNameSet: string[] = []

    for (let i = 0; i < selectedUnderlyingData.leftAggs.length; i++){
      tempLeftNameSet.push(selectedUnderlyingData.leftAggs[i].name)
    }

    for (let j = 0; j < selectedUnderlyingData.rightAggs.length; j++){
      tempRightNameSet.push(selectedUnderlyingData.rightAggs[j].name)
    }

    for (let i = 0; i < selectedUnderlyingData.leftAggs.length; i++) {
      let left = selectedUnderlyingData.leftAggs[i]
      let right: any = null
      let idxRight = tempRightNameSet.indexOf(left.name)
      if (idxRight != -1){
        right = selectedUnderlyingData.rightAggs[idxRight]

        let deltaCallsPercentNumber =
        ((right.calls - left.calls) / left.calls) * 100

        let deltaHostDurationPercentNumber =
          ((right.host_duration - left.host_duration) / left.host_duration) * 100

        let deltaSelfHostDurationPercentNumber =
          ((right.self_host_duration - left.self_host_duration) /
            left.self_host_duration) *
          100

        let deltaDeviceDurationPercentNumber =
          ((right.device_duration - left.device_duration) /
            left.device_duration) *
          100

        let deltaSelfDeviceDurationPercentNumber =
          ((right.self_device_duration - left.self_device_duration) /
            left.self_device_duration) *
          100

        tableDataSource.push({
          key: i,
          operator: left.name,
          baselineCalls: left.calls,
          expCalls: right.calls,
          deltaCalls: right.calls - left.calls,
          deltaCallsPercentNumber: deltaCallsPercentNumber,
          deltaCallsPercent: toPercentString(deltaCallsPercentNumber),

          baselineHostDuration: left.host_duration,
          expHostDuration: right.host_duration,
          deltaHostDuration: right.host_duration - left.host_duration,
          deltaHostDurationPercentNumber: deltaHostDurationPercentNumber,
          deltaHostDurationPercent: toPercentString(
            deltaHostDurationPercentNumber
          ),

          baselineSelfHostDuration: left.self_host_duration,
          expSelfHostDuration: right.self_host_duration,
          deltaSelfHostDuration:
            right.self_host_duration - left.self_host_duration,
          deltaSelfHostDurationPercentNumber: deltaSelfHostDurationPercentNumber,
          deltaSelfHostDurationPercent: toPercentString(
            deltaSelfHostDurationPercentNumber
          ),

          baselineDeviceDuration: left.device_duration,
          expDeviceDuration: right.device_duration,
          deltaDeviceDuration: right.device_duration - left.device_duration,
          deltaDeviceDurationPercentNumber: deltaDeviceDurationPercentNumber,
          deltaDeviceDurationPercent: toPercentString(
            deltaDeviceDurationPercentNumber
          ),

          baselineSelfDeviceDuration: left.self_device_duration,
          expSelfDeviceDuration: right.self_device_duration,
          deltaSelfDeviceDuration:
            right.self_device_duration - left.self_device_duration,
          deltaSelfDeviceDurationPercentNumber: deltaSelfDeviceDurationPercentNumber,
          deltaSelfDeviceDurationPercent: toPercentString(
            deltaSelfDeviceDurationPercentNumber
          )
        })
      } else {
        let deltaCallsPercentNumber =
        ((0 - left.calls) / left.calls) * 100

        let deltaKernelCallsPercentNumber =
          ((0 - left.kernel_calls) / left.kernel_calls) * 100

        let deltaHostDurationPercentNumber =
          ((0 - left.host_duration) / left.host_duration) * 100

        let deltaSelfHostDurationPercentNumber =
          ((0 - left.self_host_duration) /
            left.self_host_duration) *
          100

        let deltaDeviceDurationPercentNumber =
          ((0 - left.device_duration) /
            left.device_duration) *
          100

        let deltaSelfDeviceDurationPercentNumber =
          ((0 - left.self_device_duration) /
            left.self_device_duration) *
          100

        tableDataSource.push({
          key: i,
          operator: left.name,
          baselineCalls: left.calls,
          expCalls: 0,
          deltaCalls: 0 - left.calls,
          deltaCallsPercentNumber: deltaCallsPercentNumber,
          deltaCallsPercent: toPercentString(deltaCallsPercentNumber),

          baselineHostDuration: left.host_duration,
          expHostDuration: 0,
          deltaHostDuration: 0 - left.host_duration,
          deltaHostDurationPercentNumber: deltaHostDurationPercentNumber,
          deltaHostDurationPercent: toPercentString(
            deltaHostDurationPercentNumber
          ),

          baselineSelfHostDuration: left.self_host_duration,
          expSelfHostDuration: 0,
          deltaSelfHostDuration:
            0 - left.self_host_duration,
          deltaSelfHostDurationPercentNumber: deltaSelfHostDurationPercentNumber,
          deltaSelfHostDurationPercent: toPercentString(
            deltaSelfHostDurationPercentNumber
          ),

          baselineDeviceDuration: left.device_duration,
          expDeviceDuration: 0,
          deltaDeviceDuration: 0 - left.device_duration,
          deltaDeviceDurationPercentNumber: deltaDeviceDurationPercentNumber,
          deltaDeviceDurationPercent: toPercentString(
            deltaDeviceDurationPercentNumber
          ),

          baselineSelfDeviceDuration: left.self_device_duration,
          expSelfDeviceDuration: 0,
          deltaSelfDeviceDuration:
            0 - left.self_device_duration,
          deltaSelfDeviceDurationPercentNumber: deltaSelfDeviceDurationPercentNumber,
          deltaSelfDeviceDurationPercent: toPercentString(
            deltaSelfDeviceDurationPercentNumber
          )
        })
      }
    }

    let newIdx = selectedUnderlyingData.leftAggs.length - 1
    for (let k = 0; k < selectedUnderlyingData.rightAggs.length; k++){
      let right = selectedUnderlyingData.rightAggs[k]
      if (tempLeftNameSet.indexOf(right.name) != -1){
        continue
      }
      
      newIdx += 1

      let deltaCallsPercentNumber =
      ((right.calls - 0) / 0) * 100

      let deltaKernelCallsPercentNumber =
        ((right.kernel_calls - 0) / 0) * 100

      let deltaHostDurationPercentNumber =
        ((right.host_duration - 0) / 0) * 100

      let deltaSelfHostDurationPercentNumber =
        ((right.self_host_duration - 0) /
          0) *
        100

      let deltaDeviceDurationPercentNumber =
        ((right.device_duration - 0) /
          0) *
        100

      let deltaSelfDeviceDurationPercentNumber =
        ((right.self_device_duration - 0) /
          0) *
        100

      tableDataSource.push({
        key: newIdx,
        operator: right.name,
        baselineCalls: 0,
        expCalls: right.calls,
        deltaCalls: right.calls - 0,
        deltaCallsPercentNumber: deltaCallsPercentNumber,
        deltaCallsPercent: toPercentString(deltaCallsPercentNumber),

        baselineHostDuration: 0,
        expHostDuration: right.host_duration,
        deltaHostDuration: right.host_duration - 0,
        deltaHostDurationPercentNumber: deltaHostDurationPercentNumber,
        deltaHostDurationPercent: toPercentString(
          deltaHostDurationPercentNumber
        ),

        baselineSelfHostDuration: 0,
        expSelfHostDuration: right.self_host_duration,
        deltaSelfHostDuration:
          right.self_host_duration - 0,
        deltaSelfHostDurationPercentNumber: deltaSelfHostDurationPercentNumber,
        deltaSelfHostDurationPercent: toPercentString(
          deltaSelfHostDurationPercentNumber
        ),

        baselineDeviceDuration: 0,
        expDeviceDuration: right.device_duration,
        deltaDeviceDuration: right.device_duration - 0,
        deltaDeviceDurationPercentNumber: deltaDeviceDurationPercentNumber,
        deltaDeviceDurationPercent: toPercentString(
          deltaDeviceDurationPercentNumber
        ),

        baselineSelfDeviceDuration: 0,
        expSelfDeviceDuration: right.self_device_duration,
        deltaSelfDeviceDuration:
          right.self_device_duration - 0,
        deltaSelfDeviceDurationPercentNumber: deltaSelfDeviceDurationPercentNumber,
        deltaSelfDeviceDurationPercent: toPercentString(
          deltaSelfDeviceDurationPercentNumber
        )
      })
    }

    return tableDataSource
  }

  React.useEffect(() => {
    if (
      run.length > 0 &&
      worker.length > 0 &&
      span.length > 0 &&
      expRun.length > 0 &&
      expWorker.length > 0 &&
      expSpan.length > 0
    ) {
      setLoading(true)

      columnChartDataStack = []
      stepChartDataStack = []
      columnUnderlyingDataStack = []
      columnTableDataSourceStack = []

      api.defaultApi
        .diffnodeGet(run, worker, span, expRun, expWorker, expSpan)
        .then((resp) => {
          handleDiffNodeResp(resp)
          let rootUnderlyingData = {
            name: 'rootNode',
            path: resp.path,
            leftAggs: resp.left.aggs,
            rightAggs: resp.right.aggs
          }

          setRootUnderlyingData(rootUnderlyingData)
          let tableDataSource = generateDataSourceFromUnderlyingData(
            rootUnderlyingData!
          )
          setTableDataSource(tableDataSource)
        })
        .finally(() => setLoading(false))

      setSelectedTableColumns([...baseTableColumns, ...hostDurationColumns])
    }
  }, [run, worker, span, expRun, expWorker, expSpan])

  const handleDiffNodeResp = (resp: any) => {
    let columnChartData: any[] = []
    let stepChartData: any[] = []
    let underlyingData: ColumnUnderlyingData[] = []

    columnChartData.push([
      'Call',
      'Baseline',
      'Experiment',
      'Baseline Trend',
      'Exp Trend'
    ])
    stepChartData.push(['Call', 'Diff', 'Accumulated Diff'])

    if (resp.children.length > 0) {
      let accumulated_left_duration = 0
      let accumulated_right_duration = 0
      let accumulated_step_diff = 0
      for (let i = 0; i < resp.children.length; i++) {
        let left = resp.children[i].left
        let right = resp.children[i].right
        let currColumn: any[] = []
        let currStep: any[] = []

        let name = left.name
        if (name === COMPOSITE_NODES_NAME) {
          continue
        }

        if (name.startsWith('aten::')) {
          // Ignore aten operators
          continue
        }

        if (name.startsWith('enumerate(DataLoader)')) {
          name = name.substring(21)
        }

        if (name.startsWith('enumerate(DataPipe)')) {
          name = name.substring(19)
        }

        if (name.startsWith('nn.Module: ')) {
          name = name.substring(11)
        }

        if (name.startsWith('Optimizer.zero_grad')) {
          name = 'Optimizer.zero_grad'
        }

        if (name.startsWith('Optimizer.step')) {
          name = 'Optimizer.step'
        }

        currColumn.push(name)
        currColumn.push(left.total_duration)
        currColumn.push(right.total_duration)

        accumulated_left_duration += left.total_duration
        currColumn.push(accumulated_left_duration)

        accumulated_right_duration += right.total_duration
        currColumn.push(accumulated_right_duration)
        columnChartData.push(currColumn)

        underlyingData.push({
          name: name,
          path: resp.children[i].path,
          leftAggs: left.aggs,
          rightAggs: right.aggs
        })

        currStep.push(name)
        let stepDiff = right.total_duration - left.total_duration
        currStep.push(stepDiff)

        accumulated_step_diff += stepDiff
        currStep.push(accumulated_step_diff)

        stepChartData.push(currStep)
      }
    } else {
      let left = resp.left
      let right = resp.right
      let currColumn: any[] = []
      let currStep: any[] = []
      let name = left.name

      if (name.startsWith('nn.Module: ')) {
        name = name.substring(11)
      }

      currColumn.push(name)
      currColumn.push(left.total_duration)
      currColumn.push(right.total_duration)
      currColumn.push(left.total_duration)
      currColumn.push(right.total_duration)

      columnChartData.push(currColumn)

      currStep.push(name)
      let stepDiff = right.total_duration - left.total_duration
      currStep.push(stepDiff)
      currStep.push(stepDiff)
      stepChartData.push(currStep)
    }

    setColumnChartData(columnChartData)
    columnChartDataStack.push(columnChartData)

    setStepChartData(stepChartData)
    stepChartDataStack.push(stepChartData)

    setColumnUnderlyingData(underlyingData)
    columnUnderlyingDataStack.push(underlyingData)

    setDataStackLevel(columnChartDataStack.length)
  }

  // #endregion

  if (!loading && columnUnderlyingDataStack.length === 0) {
    return (
      <Card variant="outlined">
        <CardHeader title="No Runs Found"></CardHeader>
        <CardContent>
          <Typography>There is no run selected for diff.</Typography>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return <FullCircularProgress />
  }

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item spacing={1}>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title="DiffView" />
              <CardContent>
                <Button
                  className={classes.iconButton}
                  startIcon={<ChevronLeftIcon />}
                  onClick={handleGoBack}
                  variant="outlined"
                  disabled={dataStackLevel < 2}
                >
                  Go Back
                </Button>
                {columnChartData.length > 1 && (
                  <>
                    <DiffColumnChart
                      rawData={columnChartData}
                      selectCallback={handleChartColumnSelect}
                    />
                    <DiffStepChart rawData={stepChartData} />
                  </>
                )}
                {columnChartData.length === 1 && (
                  <Typography>No more level to show.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item spacing={1}>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title="Operator View" />
              <CardContent>
                <Select
                  mode="multiple"
                  style={{ width: '100%' }}
                  placeholder="Select the data you need"
                  value={selectedTableColumnsOptions}
                  onChange={handleColumnSelectionChange}
                  optionLabelProp="label"
                >
                  <Option value="hostDuration" label="Host Duration">
                    <div>Host Duration</div>
                  </Option>
                  <Option value="selfHostDuration" label="Self Host Duration">
                    <div>Self Host Duration</div>
                  </Option>
                  <Option value="deviceDuration" label="Device Duration">
                    <div>Device Duration</div>
                  </Option>
                  <Option
                    value="selfDeviceDuration"
                    label="Self Device Duration"
                  >
                    <div>Self Device Duration</div>
                  </Option>
                </Select>
                &nbsp;
                <Table
                  dataSource={tableDataSource}
                  columns={selectedTableColumns}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Grid>
    </div>
  )
}
