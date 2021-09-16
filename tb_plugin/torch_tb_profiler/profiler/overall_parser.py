# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from .. import utils
from .event_parser import ProfileRole
from .range_utils import (get_ranges_sum, intersection_ranges_lists,
                          merge_ranges, subtract_ranges_lists)

logger = utils.get_logger()


class OverallParser(object):
    class Costs:
        def __init__(self):
            self.costs = [0] * len(ProfileRole)

        @classmethod
        def calculate_costs(cls, statistics, step):
            cost_obj = cls()
            for i in range(len(statistics.cost_ranges)):
                cost_obj.costs[i] = get_ranges_sum(statistics.cost_ranges[i])
            cost_obj.costs[ProfileRole.Total] = step[1] - step[0]
            return cost_obj

    class Statistics:
        def __init__(self, cost_ranges):
            if not cost_ranges:
                raise ValueError("the cost ranges is None")

            self.cost_ranges = cost_ranges

        @classmethod
        def create_statistics(cls, steps, role_ranges):
            assert len(role_ranges) == ProfileRole.Total - 1

            cost_ranges = []
            slots = []
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

        def intersection_with_step(self, step):
            cost_ranges = []
            step = [step]
            for range in self.cost_ranges:
                cost_ranges.append(intersection_ranges_lists(step, range))

            return OverallParser.Statistics(cost_ranges)

    class StepCommunicationCosts:
        def __init__(self):
            self.computation = 0
            self.communication = 0
            self.overlap = 0
            self.other = 0

    def __init__(self):
        self.steps_costs = []
        self.avg_costs = OverallParser.Costs()
        self.communication_overlap = []

    def aggregate(self, steps, role_ranges):
        logger.debug("Overall, statistics")
        global_stats = OverallParser.Statistics.create_statistics(steps, role_ranges)
        if role_ranges[ProfileRole.Kernel]:
            comm_comp_overlap = intersection_ranges_lists(role_ranges[ProfileRole.Kernel], role_ranges[ProfileRole.Communication])
        else:
            comm_comp_overlap = intersection_ranges_lists(role_ranges[ProfileRole.CpuOp], role_ranges[ProfileRole.Communication])

        logger.debug("Overall, aggregation")
        valid_steps = len(steps)
        for i in range(valid_steps):
            steps_stat = global_stats.intersection_with_step(steps[i])
            self.steps_costs.append(OverallParser.Costs.calculate_costs(steps_stat, steps[i]))
            for cost_index in range(len(self.avg_costs.costs)):
                self.avg_costs.costs[cost_index] += self.steps_costs[i].costs[cost_index]

            comm_costs = OverallParser.StepCommunicationCosts()
            comm_costs.overlap = get_ranges_sum(intersection_ranges_lists([steps[i]], comm_comp_overlap))
            if role_ranges[ProfileRole.Kernel]:
                comm_costs.computation = get_ranges_sum(intersection_ranges_lists([steps[i]], role_ranges[ProfileRole.Kernel]))
            else:
                comm_costs.computation = get_ranges_sum(intersection_ranges_lists([steps[i]], role_ranges[ProfileRole.CpuOp]))
            comm_costs.communication = get_ranges_sum(intersection_ranges_lists([steps[i]], role_ranges[ProfileRole.Communication]))
            comm_costs.other = self.steps_costs[i].costs[ProfileRole.Total] + comm_costs.overlap - comm_costs.computation - comm_costs.communication
            self.communication_overlap.append(comm_costs)

        for i in range(len(self.avg_costs.costs)):
            self.avg_costs.costs[i] /= valid_steps
