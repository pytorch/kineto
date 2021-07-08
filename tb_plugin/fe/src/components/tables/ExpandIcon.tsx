/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { Button, TableProps } from 'antd'
import { OperationTableDataInner, CallStackTableDataInner } from '../../api'
import { Arguments } from '../../utils/type'

type Types<T> = NonNullable<TableProps<T>['expandable']>['expandIcon']
type BasePropType<T> = Arguments<NonNullable<Types<T>>>[0]
type PropType<T> = BasePropType<T> & { text: string; disabled?: boolean }

export function ExpandIcon<
  T extends OperationTableDataInner | CallStackTableDataInner
>(props: PropType<T>) {
  const onClick = (e: React.MouseEvent<HTMLElement, MouseEvent>) => {
    props.onExpand(props.record, e)
  }

  return (
    <Button type="link" onClick={onClick} disabled={props.disabled}>
      {props.text}
    </Button>
  )
}

export function makeExpandIcon<
  T extends OperationTableDataInner | CallStackTableDataInner
>(text: string, disabled?: (v: T) => boolean) {
  return (props: BasePropType<T>) => (
    <ExpandIcon {...props} text={text} disabled={disabled?.(props.record)} />
  )
}
