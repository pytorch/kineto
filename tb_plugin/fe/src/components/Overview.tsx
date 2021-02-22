/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import Grid from '@material-ui/core/Grid'
import Card from '@material-ui/core/Card'
import Tooltip from '@material-ui/core/Tooltip'
import CardHeader from '@material-ui/core/CardHeader'
import CardContent from '@material-ui/core/CardContent'
import { TextListItem } from './TextListItem'
import * as api from '../api'
import { DataLoading } from './DataLoading'
import { SteppedAreaChart } from './charts/SteppedAreaChart'
import {
  transformPerformanceIntoTable,
  transformPerformanceIntoPie
} from './transform'
import { TableChart } from './charts/TableChart'
import { PieChart } from './charts/PieChart'
import HelpOutline from '@material-ui/icons/HelpOutline'
import { StepTimeBreakDownTooltip } from './TooltipDescriptions'

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
  tooltip: {
    whiteSpace: 'pre-wrap',
    maxWidth: '500px'
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center'
  },
  titleText: {
    marginRight: '6px'
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
  view: string
}

export const Overview: React.FC<IProps> = (props) => {
  const { run, worker, view } = props

  const [steps, setSteps] = React.useState<api.Graph | undefined>(undefined)
  const [performances, setPerformances] = React.useState<api.Performance[]>([])
  const [environments, setEnvironments] = React.useState<api.Environment[]>([])
  const [recommendations, setRecommendations] = React.useState('')

  const synthesizedTableGraph = React.useMemo(() => {
    return transformPerformanceIntoTable(performances)
  }, [performances])

  const synthesizedPieGraph = React.useMemo(() => {
    return transformPerformanceIntoPie(performances)
  }, [performances])

  React.useEffect(() => {
    api.defaultApi.overviewGet(run, worker, view).then((resp) => {
      setPerformances(resp.performance)
      setEnvironments(resp.environments)
      setSteps(resp.steps)
      setRecommendations(resp.recommendations)
    })
  }, [run, worker, view])

  const classes = useStyles()

  const stepTimeBreakDownTitle = React.useMemo(
    () => (
      <span className={classes.cardTitle}>
        <span className={classes.titleText}>Step Time Breakdown</span>
        <Tooltip
          arrow
          classes={{ tooltip: classes.tooltip }}
          title={StepTimeBreakDownTooltip}
        >
          <HelpOutline />
        </Tooltip>
      </span>
    ),
    [classes.cardTitle, classes.tooltip]
  )

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item spacing={1}>
          <Grid item sm={4}>
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
          <Grid item sm={8}>
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
