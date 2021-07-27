# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd

from .trace import EventTypes


class KernelParser:
    def __init__(self):
        self.kernel_stat = None

    def parse_events(self, events):
        events = [vars(event) for event in events if event.type == EventTypes.KERNEL ]
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "name": "string"}, copy=False)

        def weighted_avg(x):
            try:
                # fill these None as zero
                x = x.fillna(0)
                return np.average(x, weights=events.loc[x.index, "duration"])
            except ZeroDivisionError:
                return 0

        self.kernel_stat = events.groupby("name").agg(
            count=('duration', "count"),
            sum=('duration', "sum"),
            mean=('duration', "mean"),
            max=('duration', "max"),
            min=('duration', "min"),
            blocks_per_sm=('blocks_per_sm', weighted_avg),
            occupancy=('occupancy', weighted_avg))\
            .sort_values("sum", ascending=False)
