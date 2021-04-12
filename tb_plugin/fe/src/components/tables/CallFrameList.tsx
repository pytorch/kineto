/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { CallStackFrame } from './transform'
import { List } from 'antd'
import { NavToCodeButton } from './NavToCodeButton'

interface IProps {
  callFrames: CallStackFrame[]
}

const renderItem = (item: CallStackFrame) => (
  <List.Item>
    <NavToCodeButton frame={item} />
  </List.Item>
)

export const CallFrameList = (props: IProps) => {
  return (
    <List
      pagination={false}
      size="small"
      dataSource={props.callFrames}
      renderItem={renderItem}
    />
  )
}
