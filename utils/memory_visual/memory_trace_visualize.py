# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from torch_tb_profiler.profiler.data import RunProfileData
from torch_tb_profiler.run import RunProfile

SCHEMA_VERSION = 1
WORKER_NAME = "worker0"


class MemoryTraceVisualize:
    """
    Visualize and analyze memory trace data.
    """

    def __init__(
        self,
        worker: Optional[str] = None,
    ) -> None:
        self.worker = worker if worker else WORKER_NAME
        self.df_curve = None
        self.df_stats = None

    def open_with_path(self, trace_path: str):
        with open(trace_path) as input_trace:
            self.df_stats, self.df_curve = self.process(input_trace)

    def open_with_json(self, json_input: Dict[str, Any]):
        self.df_stats, self.df_curve = self.process(json_input, True)

    def process(self, input_path_or_json, with_json=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not with_json:
            data = json.load(input_path_or_json)
            assert "traceEvents" in data, "Wrong trace file (no 'traceEvents')"
        else:
            data = input_path_or_json

        trace = json.loads(json.dumps(data["traceEvents"]))


        trace_json = {"schemaVersion": 1, "traceEvents": trace}
        profile = RunProfileData.from_json(WORKER_NAME, 0, trace_json)

        stat_results = RunProfile.get_memory_stats(profile, memory_metric="MB")
        curve_results = RunProfile.get_memory_curve(
            profile, time_metric="ms", memory_metric="MB", patch_for_step_plot=True
        )
        stat_data = stat_results["rows"][stat_results["metadata"]["default_device"]]
        h_row = [(c["name"]) for c in stat_results["columns"]]
        df_stats = (
            pd.DataFrame(stat_data, columns=h_row)
            .sort_values(by=h_row[2], ascending=False)
            .T
        )

        # default_device: GPU
        curve = curve_results["rows"][curve_results["metadata"]["default_device"]]
        df_curve = pd.DataFrame(curve).T
        return df_stats, df_curve

    def draw_curve(self, return_html_str: bool = False) -> Optional[str]:

        time, allocated, reserved = [list(sub) for sub in self.df_curve.values]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time,
                y=allocated,
                line_shape="hv",
                mode="lines+markers",
                name="allocated",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=reserved,
                line_shape="hv",
                mode="lines+markers",
                name="reserved",
            )
        )
        # Edit the layout
        fig.update_layout(  # title='Memory Trace',
            xaxis_title="Time (ms)", yaxis_title="Memory Usage (MB)"
        )

        fig.show()
        if return_html_str:
            return to_html(fig, include_plotlyjs="cdn")

    def draw_table(self, return_html_str: bool = False) -> Optional[str]:
        columns = [*self.df_stats.T]
        sort_labels = [
            ({"l": item, "c": index + 1}) for index, item in enumerate(columns[1:])
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header={"values": columns}, cells={"values": self.df_stats.values}
                )
            ]
        )
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "method": "restyle",
                            "label": b["l"],
                            "args": [
                                {
                                    "cells": {
                                        "values": self.df_stats.T.sort_values(
                                            by=columns[b["c"]], ascending=False
                                        ).T.values
                                    }
                                },
                                [0],
                            ],
                        }
                        for b in sort_labels
                    ],
                    "type": "buttons",
                    "y": 1,
                }
            ]
        )
        fig.show()
        if return_html_str:
            return to_html(fig, include_plotlyjs="cdn")
