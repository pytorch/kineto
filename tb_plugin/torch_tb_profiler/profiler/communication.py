# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from typing import Dict, List, Tuple

from .. import utils
from .node import CommunicationNode
from .range_utils import get_ranges_sum, merge_ranges

logger = utils.get_logger()


def generate_communication_nodes(
        communication_data: Dict[int, CommunicationNode],
        steps: List[Tuple[int, int]],
        steps_names: List[str]):
    comm_node_list: List[CommunicationNode] = []

    # Sort the communication node according the start time, this is for correlating communication node between workers
    for comm_node in communication_data.values():
        comm_node.kernel_ranges.sort(key=lambda x: (x[0], -x[1]))
        comm_node_list.append(comm_node)
    comm_node_list.sort(key=lambda x: (x.start_time, -x.end_time))

    # Find each communication node belong to which step
    index = 0
    valid_steps = len(steps)
    for comm_node in comm_node_list:
        while index < valid_steps:
            if comm_node.start_time >= steps[index][0] and comm_node.end_time <= steps[index][1]:
                comm_node.step_name = steps_names[index]
                break
            elif comm_node.start_time >= steps[index][1]:
                index += 1
            else:
                logger.error('Found a communication op not belong to any step.')
                break
        if index >= valid_steps:
            logger.error('Found communication ops not belong to any step. ')
            break

    return comm_node_list


def analyze_communication_nodes(comm_node_list: List[CommunicationNode])\
        -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[int]]]:
    step_comm_stats: Dict[str, Tuple[int, int]] = {}
    total_comm_stats: Dict[str, Tuple[int, int, List, List]] = {}

    step_to_comm_ranges: Dict[str, Tuple[List, List]] = {}
    for comm_node in comm_node_list:
        if comm_node.step_name not in step_to_comm_ranges:
            step_to_comm_ranges[comm_node.step_name] = [[], []]
        step_to_comm_ranges[comm_node.step_name][0].extend(comm_node.kernel_ranges)
        step_to_comm_ranges[comm_node.step_name][1].extend(comm_node.real_time_ranges)

        if comm_node.name not in total_comm_stats:
            total_comm_stats[comm_node.name] = [0, 0, [], []]
        total_comm_stats[comm_node.name][0] += 1
        bytes_one_value = 0
        if comm_node.input_shape:
            for i in range(len(comm_node.input_shape)):
                if comm_node.input_type[i] == 'long int':
                    bytes_one_value = 8
                elif comm_node.input_type[i] == 'float':
                    bytes_one_value = 4
                elif comm_node.input_type[i] == 'int':
                    bytes_one_value = 4
                elif comm_node.input_type[i] == 'c10::Half':
                    bytes_one_value = 2
                elif comm_node.input_type[i] == 'c10::BFloat16':
                    bytes_one_value = 2
                elif comm_node.input_type[i] == 'unsigned char':
                    bytes_one_value = 1
                else:
                    logger.warning('Found an unknown tensor type: {}'.format(comm_node.input_type[i]))
                    bytes_one_value = 0
                total_size = 1
                for size in comm_node.input_shape[i]:
                    total_size *= size
                total_comm_stats[comm_node.name][1] += total_size * bytes_one_value
        total_comm_stats[comm_node.name][2].extend(comm_node.kernel_ranges)
        total_comm_stats[comm_node.name][3].extend(comm_node.real_time_ranges)

    for step, comm_ranges in step_to_comm_ranges.items():
        step_comm_stats[step] = [
            get_ranges_sum(merge_ranges(comm_ranges[0])),
            get_ranges_sum(merge_ranges(comm_ranges[1]))
        ]

    for _, stats in total_comm_stats.items():
        stats[2] = get_ranges_sum(merge_ranges(stats[2]))
        stats[3] = get_ranges_sum(merge_ranges(stats[3]))

    return step_comm_stats, total_comm_stats
