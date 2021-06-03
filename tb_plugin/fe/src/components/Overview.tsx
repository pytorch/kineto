/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import * as api from '../api'
import { PieChart } from './charts/PieChart'
import { SteppedAreaChart } from './charts/SteppedAreaChart'
import { TableChart } from './charts/TableChart'
import { DataLoading } from './DataLoading'
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers'
import { TextListItem } from './TextListItem'
import { StepTimeBreakDownTooltip } from './TooltipDescriptions'
import {
  transformPerformanceIntoPie,
  transformPerformanceIntoTable
} from './transform'

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
  }
}))

const highlightNoTopLevel = (
  row: number,
  column: number,
  cb: (key: string, value: any) => void
) => {
  if (row !== 0) {
    cb('style', 'background: #e0e0e0')
  }
}

export interface IProps {
  run: string
  worker: string
  span: string
}

export const Overview: React.FC<IProps> = (props) => {
  const { run, worker, span } = props

  const [steps, setSteps] = React.useState<api.Graph | undefined>(undefined)
  const [performances, setPerformances] = React.useState<api.Performance[]>([])
  const [environments, setEnvironments] = React.useState<api.Environment[]>([])
  const [gpuMetrics, setGpuMetrics] = React.useState<
    api.GpuMetrics | undefined
  >(undefined)
  const [recommendations, setRecommendations] = React.useState('')

  const synthesizedTableGraph = React.useMemo(() => {
    return transformPerformanceIntoTable(performances)
  }, [performances])

  const synthesizedPieGraph = React.useMemo(() => {
    return transformPerformanceIntoPie(performances)
  }, [performances])

  React.useEffect(() => {
    api.defaultApi.overviewGet(run, worker, span).then((resp) => {
      setPerformances(resp.performance)
      setEnvironments(resp.environments)
      setSteps(resp.steps)
      setRecommendations(resp.recommendations)
      setGpuMetrics(resp.gpu_metrics)
      console.log(resp.gpu_metrics)
    })
  }, [run, worker, span])

  const classes = useStyles()
  const tooltipCommonClasses = useTooltipCommonStyles()
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses, false),
    [tooltipCommonClasses]
  )

  const stepTimeBreakDownTitle = React.useMemo(
    () => chartHeaderRenderer('Step Time Breakdown', StepTimeBreakDownTooltip),
    [tooltipCommonClasses, chartHeaderRenderer]
  )

  const cardSizes = gpuMetrics
    ? ([2, 3, 7] as const)
    : ([4, undefined, 8] as const)

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item spacing={1}>
          <Grid item sm={cardSizes[0]}>
            {React.useMemo(
              () => (
                <Card variant="outlined">
                  <CardHeader title="Configuration" />
                  <CardContent className={classes.topGraph}>
                    {environments.map((environment) => (
                      <TextListItem
                        name={environment.title}
                        value={environment.value}
                      />
                    ))}
                  </CardContent>
                </Card>
              ),
              [environments]
            )}
          </Grid>
          {gpuMetrics && (
            <Grid item sm={cardSizes[1]}>
              <Card variant="outlined">
                <CardHeader
                  title={chartHeaderRenderer('GPU Summary', gpuMetrics.tooltip)}
                />
                <CardContent
                  className={classes.topGraph}
                  style={{ overflow: 'auto' }}
                >
                  {gpuMetrics.data.map((metric) => (
                    <TextListItem
                      name={metric.title}
                      value={metric.value}
                      dangerouslyAllowHtml
                    />
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
          <Grid item sm={cardSizes[2]}>
            <Card variant="outlined">
              <CardHeader title="Execution Summary" />
              <CardContent>
                <Grid container spacing={1}>
                  <Grid item sm={6}>
                    <TableChart
                      graph={synthesizedTableGraph}
                      height={topGraphHeight}
                      allowHtml
                      setCellProperty={highlightNoTopLevel}
                    />
                  </Grid>
                  <Grid item sm={6}>
                    <PieChart
                      graph={synthesizedPieGraph}
                      height={topGraphHeight}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title={stepTimeBreakDownTitle} />
              <CardContent>
                <DataLoading value={steps}>
                  {(graph) => (
                    <SteppedAreaChart
                      graph={graph}
                      hAxisTitle="Step"
                      vAxisTitle={'Step Time (microseconds)'}
                    />
                  )}
                </DataLoading>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title="Performance Recommendation" />
              <CardContent>
                <div className={classes.pre}>
                  <div
                    dangerouslySetInnerHTML={{
                      __html: recommendations || 'None'
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Grid>
    </div>
  )
}
