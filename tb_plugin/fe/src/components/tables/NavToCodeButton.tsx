/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { CallStackFrame } from './transform'
import { Button } from 'antd'
import { navToCode } from '../../utils/vscode'

interface IProps {
  frame: CallStackFrame
}

export const NavToCodeButton = (props: IProps) => {
  const { raw, line, file } = props.frame
  const couldNavToFile = line && file

  const onClick = () => {
    if (line && file) {
      navToCode(file, line - 1)
    }
  }

  return (
    <Button type="link" disabled={!couldNavToFile} onClick={onClick}>
      {raw}
    </Button>
  )
}
