# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import pandas as pd


class KernelParser:
    def __init__(self):
        self.kernel_stat = None

    def parse_events(self, events):
        events_dict = []
        for event in events:
            events_dict.append(event.to_dict())
        events = events_dict
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "category": "category", "name": "string"}, copy=False)
        kernels = events[events["category"] == "Kernel"]
        self.kernel_stat = kernels.groupby("name")["duration"].agg(["count", "sum", "mean", "max", "min"]) \
            .sort_values("sum", ascending=False)
