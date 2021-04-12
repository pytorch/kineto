/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import Button from '@material-ui/core/Button'

export interface IProps {
  name: string
  inputShape?: string
  has_call_stack?: boolean
  onClick?: (name: string, inputShape?: string) => void
}

export const ViewCallStackButton = (props: IProps) => {
  const onClick = () => {
    props.onClick?.(props.name, props.inputShape)
  }

  return (
    <Button disabled={!props.has_call_stack} onClick={onClick}>
      View
    </Button>
  )
}
