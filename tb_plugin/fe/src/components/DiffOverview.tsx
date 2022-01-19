/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import Button from '@material-ui/core/Button'
import Typography from '@material-ui/core/Typography'
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'

import { makeStyles } from '@material-ui/core/styles'
import { Table } from 'antd'
import * as React from 'react'
import * as api from '../api'
import { FullCircularProgress } from './FullCircularProgress'
import { useResizeEventDependency } from '../utils/resize'

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

const DiffColumnChart: React.FC<DiffColumnChartIProps> = (props) => {
  const { rawData, selectCallback } = props
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    var options = {
      height: 500,
      seriesType: 'bars',
      series: {
        0: { type: 'bars' },
        1: { type: 'bars' },
        2: { type: 'line' },
        3: { type: 'line' }
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
  deltaCallsPercent?: string

  baselineDuration: number
  expDuration: number
  deltaDuration: number
  deltaDurationPercent: string
}

let chartDataStack: any[][] = []
let columnUnderlyingDataStack: ColumnUnderlyingData[][] = []
let columnTableDataSourceStack: TableRow[][] = []
export const DiffOverview: React.FC<IProps> = (props) => {
  const COMPOSITE_NODES_NAME = 'CompositeNodes'

  const tableColumns = [
    {
      title: 'Operator',
      dataIndex: 'operator',
      key: 'operator'
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
      key: 'deltaCalls'
    },
    {
      title: 'Delta Calls%',
      dataIndex: 'deltaCallsPercent',
      key: 'deltaCallsPercent'
    },

    {
      title: 'Baseline Duration',
      dataIndex: 'baselineDuration',
      key: 'baselineDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineDuration - b.baselineDuration
    },
    {
      title: 'Exp Duration',
      dataIndex: 'expDuration',
      key: 'expDuration',
      sorter: (a: TableRow, b: TableRow) => a.expDuration - b.expDuration
    },
    {
      title: 'Delta Duration',
      dataIndex: 'deltaDuration',
      key: 'deltaDuration'
    },
    {
      title: 'Delta Duration%',
      dataIndex: 'deltaDurationPercent',
      key: 'deltaDurationPercent'
    }
  ]

  const [tableDataSource, setTableDataSource] = React.useState<TableRow[]>([])
  const { run, worker, span, expRun, expWorker, expSpan } = props

  const [columnUnderlyingData, setColumnUnderlyingData] = React.useState<
    ColumnUnderlyingData[]
  >([])
  const [columnChartData, setColumnChartData] = React.useState<any[]>([])

  const [dataStackLevel, setDataStackLevel] = React.useState(0)
  const [loading, setLoading] = React.useState(false)

  const classes = useStyles()

  const handleChartColumnSelect = (row: number, column: number) => {
    if (columnUnderlyingData.length === 0) {
      return
    }

    let selectedUnderlyingData = columnUnderlyingData[row]
    if (!selectedUnderlyingData) {
      return
    }

    let tableDataSource: TableRow[] = []

    for (let i = 0; i < selectedUnderlyingData.leftAggs.length; i++) {
      let left = selectedUnderlyingData.leftAggs[i]
      let right = selectedUnderlyingData.rightAggs[i]

      tableDataSource.push({
        key: i,
        operator: left.name,
        baselineCalls: left.calls,
        expCalls: right.calls,
        deltaCalls: right.calls - left.calls,
        deltaCallsPercent: `${(
          ((right.calls - left.calls) / left.calls) *
          100
        ).toFixed(2)}%`,
        baselineDuration: left.self_host_duration,
        expDuration: right.self_host_duration,
        deltaDuration: right.self_host_duration - left.self_host_duration,
        deltaDurationPercent: `${(
          ((right.self_host_duration - left.self_host_duration) /
            left.self_host_duration) *
          100
        ).toFixed(2)}%`
      })
    }

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
    if (chartDataStack.length > 1) {
      chartDataStack.pop()
      let top = chartDataStack[chartDataStack.length - 1]
      setColumnChartData(top)
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
      setTableDataSource(top)
    }

    setDataStackLevel(dataStackLevel - 1)
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

      api.defaultApi
        .diffnodeGet(run, worker, span, expRun, expWorker, expSpan)
        .then((resp) => handleDiffNodeResp(resp))
        .finally(() => setLoading(false))
    }
  }, [run, worker, span, expRun, expWorker, expSpan])

  const handleDiffNodeResp = (resp: any) => {
    let data: any[] = []
    let underlyingData: ColumnUnderlyingData[] = []

    data.push(['Call', 'Baseline', 'Experiment', 'Baseline Trend', 'Exp Trend'])

    if (resp.children.length > 0) {
      for (let i = 0; i < resp.children.length; i++) {
        let left = resp.children[i].left
        let right = resp.children[i].right
        let curr: any[] = []

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

        curr.push(name)
        curr.push(left.total_duration)
        curr.push(right.total_duration)
        curr.push(left.total_duration)
        curr.push(right.total_duration)

        underlyingData.push({
          name: name,
          path: resp.children[i].path,
          leftAggs: left.aggs,
          rightAggs: right.aggs
        })

        data.push(curr)
      }
    } else {
      let left = resp.left
      let right = resp.right
      let curr: any[] = []
      let name = left.name

      if (name.startsWith('nn.Module: ')) {
        name = name.substring(11)
      }

      curr.push(name)
      curr.push(left.total_duration)
      curr.push(right.total_duration)
      curr.push(left.total_duration)
      curr.push(right.total_duration)

      data.push(curr)
    }

    setColumnChartData(data)
    chartDataStack.push(data)

    setColumnUnderlyingData(underlyingData)
    columnUnderlyingDataStack.push(underlyingData)

    setDataStackLevel(dataStackLevel + 1)
  }

  if (!loading && columnUnderlyingDataStack.length === 0) {
    return (
      <Card variant="outlined">
        <CardHeader title="No Runs Found"></CardHeader>
        <CardContent>
          <Typography>There are not any runs selected for diff.</Typography>
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
                  <DiffColumnChart
                    rawData={columnChartData}
                    selectCallback={handleChartColumnSelect}
                  />
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
                <Table dataSource={tableDataSource} columns={tableColumns} />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Grid>
    </div>
  )
}
