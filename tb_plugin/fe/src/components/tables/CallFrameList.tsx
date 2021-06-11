/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { CallStackFrame } from './transform'
import { List } from 'antd'
import { NavToCodeButton } from './NavToCodeButton'
import { makeStyles } from '@material-ui/core/styles'

interface IProps {
  callFrames: CallStackFrame[]
}

const useStyles = makeStyles(() => ({
  item: {
    paddingTop: '1px !important',
    paddingBottom: '1px !important'
  }
}))

export const CallFrameList = (props: IProps) => {
  const classes = useStyles()

  const renderItem = React.useCallback(
    (item: CallStackFrame) => (
      <List.Item className={classes.item}>
        <NavToCodeButton frame={item} />
      </List.Item>
    ),
    [classes.item]
  )

  return (
    <List
      pagination={false}
      size="small"
      dataSource={props.callFrames}
      renderItem={renderItem}
    />
  )
}
