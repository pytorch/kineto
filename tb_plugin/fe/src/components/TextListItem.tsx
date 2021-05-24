/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'

import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'

export interface IStylesProps {
  root?: string
  name?: string
}

export interface IProps {
  name: string
  value?: string
  description?: string
  extra?: string
  classes?: IStylesProps
}

const useStyles = makeStyles((theme) => ({
  label: {
    ...theme.typography.subtitle2,
    fontWeight: 'bolder'
  },
  value: {
    textAlign: 'right',
    ...theme.typography.subtitle2,
    fontWeight: 'bolder'
  }
}))

export const TextListItem: React.FC<IProps> = (props) => {
  const classes = useStyles()

  const getSizes = () => {
    if (props.value && props.extra) {
      return [4, 4, 4] as const
    }
    if (props.value) {
      if (props.value.length > props.name.length) {
        return [4, 8, undefined] as const
      }
      return [8, 4, undefined] as const
    }
    return [12, undefined, undefined] as const
  }

  const sizes = getSizes()

  return (
    <Grid container className={props.classes?.root}>
      <Grid item xs={sizes[0]}>
        <Grid container direction="column">
          <Grid item className={classes.label}>
            <span className={props.classes?.name}>{props.name}</span>
          </Grid>
          {props.description && (
            <Grid item>
              <span>{props.description}</span>
            </Grid>
          )}
        </Grid>
      </Grid>
      {props.value && (
        <Grid item xs={sizes[1]} className={classes.value}>
          <span>{props.value}</span>
        </Grid>
      )}
      {props.extra && (
        <Grid item xs={sizes[2]} className={classes.value}>
          <span>{props.extra}</span>
        </Grid>
      )}
    </Grid>
  )
}
