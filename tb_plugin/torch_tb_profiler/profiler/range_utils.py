# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from typing import List, Tuple


# src_ranges: item of (start_time, end_time, value)
def merge_ranges_with_value(src_ranges):
    from collections import namedtuple
    from enum import IntEnum

    class EndpointTypes(IntEnum):
        START = 0
        END = 1

    EndPoint = namedtuple('EndPoint', ['time', 'pt_type', 'value'])

    merged_ranges = []
    if len(src_ranges) > 0:
        # Build tuple of (time, type, value)
        endpoints: List[EndPoint] = []
        for r in src_ranges:
            endpoints.append(EndPoint(r[0], EndpointTypes.START, r[2]))
            endpoints.append(EndPoint(r[1], EndpointTypes.END, r[2]))
        endpoints.sort(key=lambda x: [x.time, int(x.pt_type)])  # Make START in front of END if equal on time.

        last_endpoint_time = endpoints[0].time
        last_value = endpoints[0].value
        for i in range(1, len(endpoints)):
            ep = endpoints[i]
            if ep.time > last_endpoint_time and last_value > 0.0:
                approximated_sm_efficiency = min(last_value, 1.0)
                merged_ranges.append((last_endpoint_time, ep.time, approximated_sm_efficiency))
            last_endpoint_time = ep.time
            if ep.pt_type == EndpointTypes.START:
                last_value += ep.value
            else:
                last_value -= ep.value

    return merged_ranges


# range_list1 item is length 3. range_list2 item is length 2.
# Reture value's item is length 3.
def intersection_ranges_lists_with_value(range_list1, range_list2) -> List[Tuple[int, int, int]]:
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
            range_list_dst.append((r1[0], r2[1], r1[2]))
            r1 = (r2[1], r1[1], r1[2])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0]:
            assert (r2[1] >= r1[1])
            range_list_dst.append(r1)
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        elif r2[1] < r1[1]:
            assert (r2[0] > r1[0])
            range_list_dst.append((r2[0], r2[1], r1[2]))
            r1 = (r2[1], r1[1], r1[2])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] < r1[1]:
            assert (r2[1] >= r1[1])
            range_list_dst.append((r2[0], r1[1], r1[2]))
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        else:
            assert (r2[0] >= r1[1])
            r1, i1 = pop_list(range_list1, i1)
    return range_list_dst


def subtract_ranges_lists(range_list1: List[Tuple[int, int]],
                          range_list2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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


def intersection_ranges_lists(range_list1: List[Tuple[int, int]],
                              range_list2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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


def get_ranges_sum(ranges: List[Tuple[int, int]]) -> int:
    sum: int = 0
    for range in ranges:
        sum += (range[1] - range[0])
    return sum


def pop_list(range_list, index):
    next_index = index + 1
    if next_index >= len(range_list):
        return None, len(range_list)
    next_item = range_list[next_index]
    return next_item, next_index


def merge_ranges(src_ranges, is_sorted=False) -> List[Tuple[int, int]]:
    if not src_ranges:
        # return empty list if src_ranges is None or its length is zero.
        return []

    if not is_sorted:
        src_ranges.sort(key=lambda x: x[0])

    merged_ranges = []
    merged_ranges.append(src_ranges[0])
    for src_id in range(1, len(src_ranges)):
        src_range = src_ranges[src_id]
        if src_range[1] > merged_ranges[-1][1]:
            if src_range[0] <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], src_range[1])
            else:
                merged_ranges.append((src_range[0], src_range[1]))

    return merged_ranges
