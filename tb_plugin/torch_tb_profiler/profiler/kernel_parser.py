# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import Optional

import numpy as np
import pandas as pd

from .tensor_core import TC_Allowlist
from .trace import EventTypes


class KernelParser:
    def __init__(self):
        self.kernel_stat: Optional[pd.DataFrame] = None
        self.tc_used_ratio = 0.0

    def parse_events(self, events):
        events = [vars(event) for event in events if event.type == EventTypes.KERNEL]
        events = pd.DataFrame(events)
        events = events.astype({'type': 'category', 'name': 'string'}, copy=False)
        events['tc_used'] = events['name'].map(lambda name: name in TC_Allowlist)

        def weighted_avg(x: pd.Series):
            try:
                # fill these None as zero
                x = x.fillna(0)
                return np.average(x, weights=events.loc[x.index, 'duration'])
            except ZeroDivisionError:
                return 0

        self.kernel_stat = events.groupby('name').agg(
            tc_used=('tc_used', 'first'),
            count=('duration', 'count'),
            sum=('duration', 'sum'),
            mean=('duration', 'mean'),
            max=('duration', 'max'),
            min=('duration', 'min'),
            blocks_per_sm=('blocks_per_sm', weighted_avg),
            occupancy=('occupancy', weighted_avg)).sort_values('sum', ascending=False)

        tc_total = self.kernel_stat['sum'].sum()
        tc_self = self.kernel_stat[self.kernel_stat['tc_used']]['sum'].sum()
        if tc_total > 0:
            self.tc_used_ratio = tc_self / tc_total
