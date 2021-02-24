import * as React from 'react'
import HelpOutline from '@material-ui/icons/HelpOutline'
import Tooltip from '@material-ui/core/Tooltip'
import { makeStyles } from '@material-ui/core/styles'
import clsx from 'clsx'

export const useTooltipCommonStyles = makeStyles((theme) => ({
  tooltip: {
    maxWidth: '600px',
    whiteSpace: 'pre-wrap',
    fontSize: '14px'
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center'
  },
  titleText: {
    marginRight: theme.spacing(0.5)
  },
  smallTitleText: {
    fontSize: '.8rem',
    fontWeight: 'bold'
  }
}))

export const makeChartHeaderRenderer = (
  classes: ReturnType<typeof useTooltipCommonStyles>,
  smallTitleText = true
) => (title: string, tooltip: string) => {
  return (
    <span className={classes.cardTitle}>
      <span
        className={clsx(
          classes.titleText,
          smallTitleText && classes.smallTitleText
        )}
      >
        {title}
      </span>
      <Tooltip arrow classes={{ tooltip: classes.tooltip }} title={tooltip}>
        <HelpOutline fontSize={smallTitleText ? 'small' : undefined} />
      </Tooltip>
    </span>
  )
}
