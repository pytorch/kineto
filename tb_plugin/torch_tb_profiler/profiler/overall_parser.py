# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from collections import defaultdict

from .. import utils
from .event_parser import ProfileRole
from .range_utils import (get_ranges_sum, intersection_ranges_lists,
                          merge_ranges, subtract_ranges_lists)

logger = utils.get_logger()


class OverallParser(object):
    class Costs:
        def __init__(self):
            self.costs = defaultdict(float)

        @classmethod
        def calculate_costs(cls, statistics, step):
            cost_obj = cls()
            for role in statistics.cost_ranges:
                cost_obj.costs[role] = get_ranges_sum(statistics.cost_ranges[role])
            cost_obj.costs[ProfileRole.Total] = step[1] - step[0]
            return cost_obj

    class Statistics:
        def __init__(self, cost_ranges):
            assert isinstance(cost_ranges, (dict, defaultdict)), type(cost_ranges)
            if not cost_ranges:
                raise ValueError("the cost ranges is None")

            self.cost_ranges = cost_ranges

        @classmethod
        def create_statistics(cls, steps, role_ranges):
            assert len(role_ranges) == len(ProfileRole) - 1, "ProfileRole.Total should be absent from role_ranges"

            cost_ranges = defaultdict(list)
            slots = []
            for role in role_ranges:
                if slots:
                    range = intersection_ranges_lists(slots, role_ranges[role])
                else:
                    range = role_ranges[role]
                    slots = merge_ranges(list(steps))
                cost_ranges[role] = range
                slots = subtract_ranges_lists(slots, range)
            cost_ranges[ProfileRole.Other] = slots

            return cls(cost_ranges)

        def intersection_with_step(self, step):
            cost_ranges = defaultdict(list)
            step = [step]
            for role in self.cost_ranges:
                cost_ranges[role] = intersection_ranges_lists(step, self.cost_ranges[role])

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
        comm_kernel_overlap = intersection_ranges_lists(role_ranges[ProfileRole.Kernel], role_ranges[ProfileRole.Communication])

        logger.debug("Overall, aggregation")
        valid_steps = len(steps)
        for i in range(valid_steps):
            steps_stat = global_stats.intersection_with_step(steps[i])
            self.steps_costs.append(OverallParser.Costs.calculate_costs(steps_stat, steps[i]))
            for role in self.steps_costs[i].costs:
                self.avg_costs.costs[role] += self.steps_costs[i].costs[role]

            comm_costs = OverallParser.StepCommunicationCosts()
            comm_costs.overlap = get_ranges_sum(intersection_ranges_lists([steps[i]], comm_kernel_overlap))
            comm_costs.computation = get_ranges_sum(intersection_ranges_lists([steps[i]], role_ranges[ProfileRole.Kernel]))
            comm_costs.communication = get_ranges_sum(intersection_ranges_lists([steps[i]], role_ranges[ProfileRole.Communication]))
            comm_costs.other = self.steps_costs[i].costs[ProfileRole.Total] + comm_costs.overlap - comm_costs.computation - comm_costs.communication
            self.communication_overlap.append(comm_costs)

        for role in self.avg_costs.costs:
            self.avg_costs.costs[role] /= valid_steps
