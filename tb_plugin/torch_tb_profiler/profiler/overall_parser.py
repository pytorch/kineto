# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from .. import utils
from .step_parser import ProfileRole, merge_ranges

logger = utils.get_logger()


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

    def aggregate(self, node_parser, step_parser):
        logger.debug("Overall, statistics")
        global_stats = OverallParser.Statistics.create_statistics(step_parser.steps, step_parser.role_ranges)
        comm_kernel_overlap = intersection_ranges_lists(step_parser.role_ranges[ProfileRole.Kernel], step_parser.role_ranges[ProfileRole.Communication])

        logger.debug("Overall, aggregation")
        valid_steps = len(step_parser.steps)
        for i in range(valid_steps):
            steps_stat = global_stats.intersection_with_step(step_parser.steps[i])
            self.steps_costs.append(OverallParser.Costs.calculate_costs(steps_stat, step_parser.steps[i]))
            for cost_index in range(len(self.avg_costs.costs)):
                self.avg_costs.costs[cost_index] += self.steps_costs[i].costs[cost_index]

            comm_costs = OverallParser.StepCommunicationCosts()
            comm_costs.overlap = get_ranges_sum(intersection_ranges_lists([step_parser.steps[i]], comm_kernel_overlap))
            comm_costs.computation = get_ranges_sum(intersection_ranges_lists([step_parser.steps[i]], step_parser.role_ranges[ProfileRole.Kernel]))
            comm_costs.communication = get_ranges_sum(intersection_ranges_lists([step_parser.steps[i]], step_parser.role_ranges[ProfileRole.Communication]))
            comm_costs.other = self.steps_costs[i].costs[ProfileRole.Total] + comm_costs.overlap - comm_costs.computation - comm_costs.communication
            self.communication_overlap.append(comm_costs)

        for i in range(len(self.avg_costs.costs)):
            self.avg_costs.costs[i] /= valid_steps

        # Find each communication node belong to which step
        index = 0
        for comm_node in node_parser.comm_node_list:
            while index < valid_steps:
                if comm_node.start_time >= step_parser.steps[index][0] and comm_node.end_time <= step_parser.steps[index][1]:
                    comm_node.step_name = step_parser.steps_names[index]
                    break
                elif comm_node.start_time >= step_parser.steps[index][1]:
                    index += 1
                else:
                    logger.error("Found a communication op not belong to any step.")
                    break
            if index >= valid_steps:
                logger.error("Found communication ops not belong to any step. ")
                break
