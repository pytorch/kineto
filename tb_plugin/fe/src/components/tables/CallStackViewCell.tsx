/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import Popover from '@material-ui/core/Popover'
import Button from '@material-ui/core/Button'
import { CallStackFrame } from './transform'
import { NavToCodeButton } from './NavToCodeButton'

interface IProps {
  frames: CallStackFrame[]
}

export const CallStackViewCell = (props: IProps) => {
  const { frames } = props
  const [popoverOpen, setPopoverOpen] = React.useState(false)

  const onPopoverOpen = () => {
    setPopoverOpen(true)
  }

  const opPopoverClose = () => {
    setPopoverOpen(false)
  }

  return (
    <div>
      <Button onClick={onPopoverOpen}>View Call Stack</Button>
      <Popover open={popoverOpen} onClose={opPopoverClose}>
        <ul>
          {frames.map((frame, i) => (
            <li key={i}>
              <NavToCodeButton frame={frame} />
            </li>
          ))}
        </ul>
      </Popover>
    </div>
  )
}
