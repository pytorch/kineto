# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd


class KernelParser:
    def __init__(self):
        self.kernel_stat = None

    def parse_events(self, events):
        events_dict = []
        for event in events:
            events_dict.append(vars(event))
            if event.category == "Kernel":
                events_dict[-1]["blocks_per_sm"] = event.args.get("blocks per SM", 0)
                events_dict[-1]["occupancy"] = event.args.get("est. achieved occupancy %", 0)
        events = events_dict
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "category": "category", "name": "string"}, copy=False)
        kernels = events[events["category"] == "Kernel"]
        weighted_avg = lambda x: np.average(x, weights=kernels.loc[x.index, "duration"])
        # remove 0 duration kernels in case of divided by zero.
        self.kernel_stat = kernels.loc[kernels["duration"] > 0].groupby("name").agg(
            count=('duration', "count"),
            sum=('duration', "sum"),
            mean=('duration', "mean"),
            max=('duration', "max"),
            min=('duration', "min"),
            blocks_per_sm=('blocks_per_sm', weighted_avg),
            occupancy=('occupancy', weighted_avg))
