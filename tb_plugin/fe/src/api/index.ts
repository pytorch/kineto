/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as api from './generated'
import { MockAPI } from './mock'

export const defaultApi = new api.DefaultApi(undefined, undefined, fetch)
export const mockApi = new MockAPI()
export * from './generated/api'
