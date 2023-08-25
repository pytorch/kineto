# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import List, Tuple

from .. import utils
from .event_parser import ProfileRole
from .range_utils import (get_ranges_sum, intersection_ranges_lists,
                          merge_ranges, subtract_ranges_lists)

logger = utils.get_logger()


class OverallParser:
    class Costs:
        def __init__(self, costs: List[float] = None):
            # the cost length is len(ProfileRole)
            if costs is None:
                self.costs = [0.] * len(ProfileRole)
            else:
                self.costs = costs

        @classmethod
        def create_from_statistics(cls, statistics: 'OverallParser.Statistics', total_duration: int):
            costs = [0.] * len(ProfileRole)
            for i in range(len(statistics.cost_ranges)):
                costs[i] = get_ranges_sum(statistics.cost_ranges[i])
            costs[ProfileRole.Total] = total_duration
            return cls(costs)

    class Statistics:
        def __init__(self, cost_ranges: List[List[Tuple[int, int]]]):
            if not cost_ranges:
                raise ValueError('the cost ranges is None')

            self.cost_ranges = cost_ranges

        @classmethod
        def create_from_range(cls, steps: List[Tuple[int, int]], role_ranges: List[List[Tuple[int, int]]]):
            assert len(role_ranges) == ProfileRole.Total - 1

            cost_ranges: List[List[Tuple[int, int]]] = []
            slots: List[Tuple[int, int]] = []
            for role in role_ranges:
                if slots:
                    range = intersection_ranges_lists(slots, role)
                else:
                    range = role
                    slots = merge_ranges(list(steps))
                cost_ranges.append(range)
                slots = subtract_ranges_lists(slots, range)
            # The last one is ProfileRole.Other
            cost_ranges.append(slots)

            return cls(cost_ranges)

        def intersection_with_step(self, step: Tuple[int, int]):
            cost_ranges: List[List[Tuple[int, int]]] = []
            step = [step]
            for range in self.cost_ranges:
                cost_ranges.append(intersection_ranges_lists(step, range))

            return OverallParser.Statistics(cost_ranges)

    class StepCommunicationCosts:
        def __init__(self):
            self.computation: int = 0
            self.communication: int = 0
            self.overlap: int = 0
            self.other: int = 0

    def __init__(self):
        self.steps_costs: List[OverallParser.Costs] = []
        self.avg_costs = OverallParser.Costs()
        self.communication_overlap: List[OverallParser.StepCommunicationCosts] = []

    def aggregate(self, steps: List[Tuple[int, int]], role_ranges: List[List[Tuple[int, int]]]):
        logger.debug('Overall, statistics')
        global_stats = OverallParser.Statistics.create_from_range(steps, role_ranges)
        if role_ranges[ProfileRole.Kernel]:
            comm_comp_overlap = intersection_ranges_lists(
                role_ranges[ProfileRole.Kernel], role_ranges[ProfileRole.Communication])
        else:
            comm_comp_overlap = intersection_ranges_lists(
                role_ranges[ProfileRole.CpuOp], role_ranges[ProfileRole.Communication])

        logger.debug('Overall, aggregation')
        for i, step in enumerate(steps):
            steps_stat = global_stats.intersection_with_step(step)
            self.steps_costs.append(OverallParser.Costs.create_from_statistics(steps_stat, step[1] - step[0]))
            for cost_index in range(len(self.avg_costs.costs)):
                self.avg_costs.costs[cost_index] += self.steps_costs[i].costs[cost_index]

            comm_costs = OverallParser.StepCommunicationCosts()
            comm_costs.overlap = get_ranges_sum(intersection_ranges_lists([step], comm_comp_overlap))
            if role_ranges[ProfileRole.Kernel]:
                comm_costs.computation = get_ranges_sum(
                    intersection_ranges_lists([step], role_ranges[ProfileRole.Kernel]))
            else:
                comm_costs.computation = get_ranges_sum(
                    intersection_ranges_lists([step], role_ranges[ProfileRole.CpuOp]))
            comm_costs.communication = get_ranges_sum(
                intersection_ranges_lists([step], role_ranges[ProfileRole.Communication]))
            comm_costs.other = self.steps_costs[i].costs[ProfileRole.Total] +\
                comm_costs.overlap - comm_costs.computation - comm_costs.communication
            self.communication_overlap.append(comm_costs)

        valid_steps = len(steps)
        for i in range(len(self.avg_costs.costs)):
            self.avg_costs.costs[i] /= valid_steps
