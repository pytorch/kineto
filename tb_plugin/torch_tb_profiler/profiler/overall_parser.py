# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import sys

from .. import utils
from .trace import EventTypes

logger = utils.get_logger()


def merge_ranges(src_ranges, is_sorted=False):
    merged_ranges = []
    if len(src_ranges) > 0:
        if not is_sorted:
            src_ranges.sort(key=lambda x: x[0])
        src_id = 0
        merged_ranges.append(
            (src_ranges[src_id][0], src_ranges[src_id][1]))
        for src_id in range(1, len(src_ranges)):
            dst_id = len(merged_ranges) - 1
            if src_ranges[src_id][1] > merged_ranges[dst_id][1]:
                if src_ranges[src_id][0] <= merged_ranges[dst_id][1]:
                    merged_ranges[dst_id] = (merged_ranges[dst_id][0], src_ranges[src_id][1])
                else:
                    merged_ranges.append(
                        (src_ranges[src_id][0], src_ranges[src_id][1]))
    return merged_ranges


def subtract_ranges_lists(range_list1, range_list2):
    range_list_dst = []
    if len(range_list1) == 0:
        return range_list_dst
    if len(range_list2) == 0:
        range_list_dst = list(range_list1)
        return range_list_dst
    r1 = range_list1[0]
    r2 = range_list2[0]
    i1 = i2 = 0
    while i1 < len(range_list1):
        if i2 == len(range_list2):
            range_list_dst.append(r1)
            r1, i1 = pop_list(range_list1, i1)
        elif r2[1] <= r1[0]:
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0] and r2[1] < r1[1]:
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0]:
            assert (r2[1] >= r1[1])
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        elif r2[0] < r1[1]:
            assert (r2[0] > r1[0])
            range_list_dst.append((r1[0], r2[0]))
            r1 = (r2[0], r1[1])
        else:
            assert (r2[0] >= r1[1])
            range_list_dst.append(r1)
            r1, i1 = pop_list(range_list1, i1)
    return range_list_dst


def intersection_ranges_lists(range_list1, range_list2):
    range_list_dst = []
    if len(range_list1) == 0 or len(range_list2) == 0:
        return range_list_dst
    r1 = range_list1[0]
    r2 = range_list2[0]
    i1 = i2 = 0
    while i1 < len(range_list1):
        if i2 == len(range_list2):
            break
        elif r2[1] <= r1[0]:
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0] and r2[1] < r1[1]:
            assert (r2[1] > r1[0])
            range_list_dst.append((r1[0], r2[1]))
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0]:
            assert (r2[1] >= r1[1])
            range_list_dst.append(r1)
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        elif r2[1] < r1[1]:
            assert (r2[0] > r1[0])
            range_list_dst.append(r2)
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] < r1[1]:
            assert (r2[1] >= r1[1])
            range_list_dst.append((r2[0], r1[1]))
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        else:
            assert (r2[0] >= r1[1])
            r1, i1 = pop_list(range_list1, i1)
    return range_list_dst


def get_ranges_sum(ranges):
    sum = 0
    for range in ranges:
        sum += (range[1] - range[0])
    return sum


def pop_list(range_list, index):
    next_index = index + 1
    if next_index >= len(range_list):
        return None, len(range_list)
    next_item = range_list[next_index]
    return next_item, next_index


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

        def calculate_costs(self, statistics, step):
            self.step_total_cost = step[1] - step[0]
            self.kernel_cost = get_ranges_sum(statistics.kernel_cost_ranges)
            self.memcpy_cost = get_ranges_sum(statistics.memcpy_cost_ranges)
            self.memset_cost = get_ranges_sum(statistics.memset_cost_ranges)
            self.runtime_cost = get_ranges_sum(statistics.runtime_cost_ranges)
            self.dataloader_cost = get_ranges_sum(statistics.dataloader_cost_ranges)
            self.cpuop_cost = get_ranges_sum(statistics.cpuop_cost_ranges)
            self.other_cost = get_ranges_sum(statistics.other_cost_ranges)

    class Statistics:
        def __init__(self):
            self.kernel_cost_ranges = []
            self.memcpy_cost_ranges = []
            self.memset_cost_ranges = []
            self.runtime_cost_ranges = []
            self.dataloader_cost_ranges = []
            self.cpuop_cost_ranges = []
            self.other_cost_ranges = []

        def intersection_with_step(self, step):
            result = OverallParser.Statistics()
            step = [step]
            result.kernel_cost_ranges = intersection_ranges_lists(step, self.kernel_cost_ranges)
            result.memcpy_cost_ranges = intersection_ranges_lists(step, self.memcpy_cost_ranges)
            result.memset_cost_ranges = intersection_ranges_lists(step, self.memset_cost_ranges)
            result.runtime_cost_ranges = intersection_ranges_lists(step, self.runtime_cost_ranges)
            result.dataloader_cost_ranges = intersection_ranges_lists(step, self.dataloader_cost_ranges)
            result.cpuop_cost_ranges = intersection_ranges_lists(step, self.cpuop_cost_ranges)
            result.other_cost_ranges = intersection_ranges_lists(step, self.other_cost_ranges)
            return result

    def __init__(self):
        self.kernel_ranges = []
        self.memcpy_ranges = []
        self.memset_ranges = []
        self.runtime_ranges = []
        self.dataloader_ranges = []
        self.cpuop_ranges = []
        self.steps = []
        self.steps_names = []
        self.has_runtime = False
        self.has_kernel = False
        self.has_memcpy_or_memset = False
        self.min_ts = sys.maxsize
        self.max_ts = -sys.maxsize - 1
        self.steps_costs = []
        self.avg_costs = OverallParser.Costs()

    def parse_events(self, events):
        logger.debug("Overall, parse events")
        for event in events:
            self.parse_event(event)

        self.kernel_ranges = merge_ranges(self.kernel_ranges)
        self.memcpy_ranges = merge_ranges(self.memcpy_ranges)
        self.memset_ranges = merge_ranges(self.memset_ranges)
        self.runtime_ranges = merge_ranges(self.runtime_ranges)
        self.dataloader_ranges = merge_ranges(self.dataloader_ranges)
        self.cpuop_ranges = merge_ranges(self.cpuop_ranges)
        if len(self.steps) == 0:
            self.steps.append((self.min_ts, self.max_ts))
            self.steps_names.append("0")
        merged_steps = list(self.steps)
        merged_steps = merge_ranges(merged_steps)

        logger.debug("Overall, statistics")
        global_stats = OverallParser.Statistics()
        global_stats.kernel_cost_ranges = self.kernel_ranges
        slots = subtract_ranges_lists(merged_steps, self.kernel_ranges)
        global_stats.memcpy_cost_ranges = intersection_ranges_lists(slots, self.memcpy_ranges)
        slots = subtract_ranges_lists(slots, global_stats.memcpy_cost_ranges)
        global_stats.memset_cost_ranges = intersection_ranges_lists(slots, self.memset_ranges)
        slots = subtract_ranges_lists(slots, global_stats.memset_cost_ranges)
        global_stats.runtime_cost_ranges = intersection_ranges_lists(slots, self.runtime_ranges)
        slots = subtract_ranges_lists(slots, global_stats.runtime_cost_ranges)
        global_stats.dataloader_cost_ranges = intersection_ranges_lists(slots, self.dataloader_ranges)
        slots = subtract_ranges_lists(slots, global_stats.dataloader_cost_ranges)
        global_stats.cpuop_cost_ranges = intersection_ranges_lists(slots, self.cpuop_ranges)
        slots = subtract_ranges_lists(slots, global_stats.cpuop_cost_ranges)
        global_stats.other_cost_ranges = slots

        logger.debug("Overall, aggregation")
        valid_steps = len(self.steps)
        for i in range(valid_steps):
            steps_stat = global_stats.intersection_with_step(self.steps[i])
            self.steps_costs.append(OverallParser.Costs())
            self.steps_costs[i].calculate_costs(steps_stat, self.steps[i])
            self.avg_costs.step_total_cost += self.steps_costs[i].step_total_cost
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
            self.kernel_ranges.append((ts, ts + dur))
            self.has_kernel = True
        elif evt_type == EventTypes.MEMCPY:
            self.memcpy_ranges.append((ts, ts + dur))
            self.has_memcpy_or_memset = True
        elif evt_type == EventTypes.MEMSET:
            self.memset_ranges.append((ts, ts + dur))
            self.has_memcpy_or_memset = True
        elif evt_type == EventTypes.RUNTIME:
            self.runtime_ranges.append((ts, ts + dur))
            self.has_runtime = True
        elif evt_type == EventTypes.OPERATOR and event.name.startswith("enumerate(DataLoader)#") \
                and event.name.endswith(".__next__"):
            self.dataloader_ranges.append((ts, ts + dur))
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR]:
            self.cpuop_ranges.append((ts, ts + dur))

        if ts < self.min_ts:
            self.min_ts = ts
        if ts + dur > self.max_ts:
            self.max_ts = ts + dur
