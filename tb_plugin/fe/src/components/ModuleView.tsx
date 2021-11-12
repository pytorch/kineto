/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/
import Card from '@material-ui/core/Card'
import CardHeader from '@material-ui/core/CardHeader'
import { makeStyles } from '@material-ui/core/styles'
import { Table } from 'antd'
import * as React from 'react'
import * as api from '../api'
import { ModuleStats } from '../api'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

const getKeyedTableColumns = function (columns: any) {
  return columns.map(function (col: any) {
    return {
      dataIndex: col.key,
      key: col.key,
      title: col.name
    }
  })
}

const getTableRows = function (key: number, rows: any) {
  return rows.map(function (row: any) {
    return {
      key: key++,
      name: row.name,
      occurences: row.occurences,
      operators: row.operators,
      host_duration: row.host_duration,
      self_host_duration: row.self_host_duration,
      device_duration: row.device_duration,
      self_device_duration: row.self_device_duration,
      children: getTableRows(key, row.children)
    }
  })
}

export const ModuleView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [moduleView, setModuleView] = React.useState<ModuleStats | undefined>(
    undefined
  )

  const rows = React.useMemo(() => {
    if (moduleView) {
      return getTableRows(1, moduleView.data)
    } else {
      return undefined
    }
  }, [moduleView])

  const columns = React.useMemo(() => {
    if (moduleView) {
      return getKeyedTableColumns(moduleView.columns)
    } else {
      return undefined
    }
  }, [moduleView])
  React.useEffect(() => {
    api.defaultApi.moduleGet(run, worker, span).then((resp) => {
      setModuleView(resp)
      console.log('module data:')
      console.log(resp)
    })
  }, [run, worker, span])

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Module View" />

        <Table
          size="small"
          bordered
          columns={columns}
          dataSource={rows}
          expandable={{
            defaultExpandAllRows: true
          }}
        />
      </Card>
    </div>
  )
}
