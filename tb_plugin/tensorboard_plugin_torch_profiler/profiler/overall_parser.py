# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import sys

import portion as P

from .. import utils
from .trace import EventTypes

logger = utils.get_logger()


def get_ranges_sum(ranges):
    sum = 0
    for item in ranges:
        if not item.empty:
            sum += item.upper - item.lower
    return sum


class OverallParser(object):
    class Costs:
        def __init__(self):
            self.step_total_cost = 0
            self.kernel_cost = 0
            self.memcpy_cost = 0
            self.memset_cost = 0
            self.runtime_cost = 0
            self.dataloader_cost = 0
            self.cpuop_cost = 0
            self.other_cost = 0
            self.steps_stat = None

    class Statistics:
        def __init__(self):
            self.kernel_cost_ranges = P.empty()
            self.memcpy_cost_ranges = P.empty()
            self.memset_cost_ranges = P.empty()
            self.runtime_cost_ranges = P.empty()
            self.dataloader_cost_ranges = P.empty()
            self.cpuop_cost_ranges = P.empty()
            self.other_cost_ranges = P.empty()

        def intersection_with_step(self, step):
            result = OverallParser.Statistics()
            step = P.closed(step[0], step[1])
            result.kernel_cost_ranges = step & self.kernel_cost_ranges
            result.memcpy_cost_ranges = step & self.memcpy_cost_ranges
            result.memset_cost_ranges = step & self.memset_cost_ranges
            result.runtime_cost_ranges = step & self.runtime_cost_ranges
            result.dataloader_cost_ranges = step & self.dataloader_cost_ranges
            result.cpuop_cost_ranges = step & self.cpuop_cost_ranges
            result.other_cost_ranges = step & self.other_cost_ranges
            return result

        def get_costs(self):
            result = OverallParser.Costs()
            result.kernel_cost = get_ranges_sum(self.kernel_cost_ranges)
            result.memcpy_cost = get_ranges_sum(self.memcpy_cost_ranges)
            result.memset_cost = get_ranges_sum(self.memset_cost_ranges)
            result.runtime_cost = get_ranges_sum(self.runtime_cost_ranges)
            result.dataloader_cost = get_ranges_sum(self.dataloader_cost_ranges)
            result.cpuop_cost = get_ranges_sum(self.cpuop_cost_ranges)
            result.other_cost = get_ranges_sum(self.other_cost_ranges)
            return result

    def __init__(self):
        self.kernel_ranges = []
        self.memcpy_ranges = []
        self.memset_ranges = []
        self.runtime_ranges = []
        self.dataloader_ranges = []
        self.cpuop_ranges = []
        self.merged_steps = []
        self.steps = []
        self.steps_names = []
        self.is_gpu_used = False
        self.min_ts = sys.maxsize
        self.max_ts = -sys.maxsize - 1
        self.steps_costs = []
        self.avg_costs = OverallParser.Costs()

    def parse_events(self, events):
        logger.debug("Overall, parse events")
        for event in events:
            self.parse_event(event)

        self.kernel_ranges = P.Interval(*self.kernel_ranges)
        self.memcpy_ranges = P.Interval(*self.memcpy_ranges)
        self.memset_ranges = P.Interval(*self.memset_ranges)
        self.runtime_ranges = P.Interval(*self.runtime_ranges)
        self.dataloader_ranges = P.Interval(*self.dataloader_ranges)
        self.cpuop_ranges = P.Interval(*self.cpuop_ranges)
        self.merged_steps = P.Interval(*self.merged_steps)

        if len(self.steps) == 0:
            self.steps.append((self.min_ts, self.max_ts))
            self.merged_steps = P.closed(self.min_ts, self.max_ts)
            self.steps_names.append("0")

        logger.debug("Overall, statistics")
        global_stats = OverallParser.Statistics()
        global_stats.kernel_cost_ranges = self.kernel_ranges
        slots = self.merged_steps - self.kernel_ranges
        global_stats.memcpy_cost_ranges = slots & self.memcpy_ranges
        slots = slots - global_stats.memcpy_cost_ranges
        global_stats.memset_cost_ranges = slots & self.memset_ranges
        slots = slots - global_stats.memset_cost_ranges
        global_stats.runtime_cost_ranges = slots & self.runtime_ranges
        slots = slots - global_stats.runtime_cost_ranges
        global_stats.dataloader_cost_ranges = slots & self.dataloader_ranges
        slots = slots - global_stats.dataloader_cost_ranges
        global_stats.cpuop_cost_ranges = slots & self.cpuop_ranges
        slots = slots - global_stats.cpuop_cost_ranges
        global_stats.other_cost_ranges = slots

        logger.debug("Overall, aggregation")
        valid_steps = len(self.steps)
        self.steps_costs = []
        for i in range(valid_steps):
            self.steps_stat = global_stats.intersection_with_step(self.steps[i])
            self.steps_costs.append(self.steps_stat.get_costs())
            self.avg_costs.step_total_cost += (self.steps[i][1] - self.steps[i][0])
            self.avg_costs.kernel_cost += self.steps_costs[i].kernel_cost
            self.avg_costs.memcpy_cost += self.steps_costs[i].memcpy_cost
            self.avg_costs.memset_cost += self.steps_costs[i].memset_cost
            self.avg_costs.runtime_cost += self.steps_costs[i].runtime_cost
            self.avg_costs.dataloader_cost += self.steps_costs[i].dataloader_cost
            self.avg_costs.cpuop_cost += self.steps_costs[i].cpuop_cost
            self.avg_costs.other_cost += self.steps_costs[i].other_cost

        self.avg_costs.step_total_cost /= valid_steps
        self.avg_costs.kernel_cost /= valid_steps
        self.avg_costs.memcpy_cost /= valid_steps
        self.avg_costs.memset_cost /= valid_steps
        self.avg_costs.runtime_cost /= valid_steps
        self.avg_costs.dataloader_cost /= valid_steps
        self.avg_costs.cpuop_cost /= valid_steps
        self.avg_costs.other_cost /= valid_steps

    def parse_event(self, event):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            self.kernel_ranges.append(P.closed(ts, ts + dur))
        elif evt_type == EventTypes.MEMCPY:
            self.memcpy_ranges.append(P.closed(ts, ts + dur))
        elif evt_type == EventTypes.MEMSET:
            self.memset_ranges.append(P.closed(ts, ts + dur))
        elif evt_type == EventTypes.RUNTIME:
            self.runtime_ranges.append(P.closed(ts, ts + dur))
        elif evt_type == EventTypes.OPERATOR and event.name.startswith("enumerate(DataLoader)#") \
                and event.name.endswith(".__next__"):
            self.dataloader_ranges.append(P.closed(ts, ts + dur))
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.merged_steps.append(P.closed(ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR]:
            self.cpuop_ranges.append(P.closed(ts, ts + dur))

        if evt_type == EventTypes.RUNTIME:
            self.is_gpu_used = True

        if ts < self.min_ts:
            self.min_ts = ts
        if ts + dur > self.max_ts:
            self.max_ts = ts + dur
