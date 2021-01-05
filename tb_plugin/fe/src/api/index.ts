/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as api from './generated'

export const defaultApi = new api.DefaultApi(undefined, undefined, fetch)
export * from './generated/api'
