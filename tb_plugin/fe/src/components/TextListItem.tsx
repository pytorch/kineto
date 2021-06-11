/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'

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
  dangerouslyAllowHtml?: boolean
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

  const getSizes = function () {
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

  const renderSpan = function (content: string, className?: string) {
    if (props.dangerouslyAllowHtml) {
      return (
        <span
          className={className}
          dangerouslySetInnerHTML={{ __html: content }}
        />
      )
    }
    return <span className={className}>{content}</span>
  }

  return (
    <Grid container className={props.classes?.root}>
      <Grid item xs={sizes[0]}>
        <Grid container direction="column">
          <Grid item className={classes.label}>
            {renderSpan(props.name, props.classes?.name)}
          </Grid>
          {props.description && (
            <Grid item>{renderSpan(props.description)}</Grid>
          )}
        </Grid>
      </Grid>
      {props.value && (
        <Grid item xs={sizes[1]} className={classes.value}>
          {renderSpan(props.value)}
        </Grid>
      )}
      {props.extra && (
        <Grid item xs={sizes[2]} className={classes.value}>
          {renderSpan(props.extra)}
        </Grid>
      )}
    </Grid>
  )
}
