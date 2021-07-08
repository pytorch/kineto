/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/
import CircularProgress from '@material-ui/core/CircularProgress'
import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'

const useStyles = makeStyles(() => ({
  root: {
    width: '100%',
    display: 'flex',
    justifyContent: 'center'
  }
}))

export const FullCircularProgress: React.FC = () => {
  const classes = useStyles()
  return (
    <div className={classes.root}>
      <CircularProgress />
    </div>
  )
}
