import unittest
import math

from torch_tb_profiler.profiler.overall_parser import (
    merge_ranges, subtract_ranges_lists, intersection_ranges_lists, get_ranges_sum
)


def check_ranges_equal(ranges1, ranges2):
    if len(ranges1) != len(ranges2):
        return False
    for i in range(len(ranges1)):
        if ranges1[i][0] != ranges2[i][0] or ranges1[i][1] != ranges2[i][1]:
            return False
    return True


class TestOverallParser(unittest.TestCase):
    def test_merge_ranges(self):
        src_ranges = [(1.1, 2.2), (1.5, 2.3), (3.3, 3.9), (3.5, 3.6), (3.7, 3.8), (4.1, 4.2)]
        expected_ranges = [(1.1, 2.3), (3.3, 3.9), (4.1, 4.2)]
        dst_ranges = merge_ranges(src_ranges, True)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_subtract_ranges_lists(self):
        ranges1 = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        ranges2 = [(0, 0.1), (1.0, 1.4), (1.5, 1.6), (1.9, 3.4), (4.3, 4.6)]
        expected_ranges = [(1.4, 1.5), (1.6, 1.9), (3.4, 4.3), (5.5, 6.6)]
        dst_ranges = subtract_ranges_lists(ranges1, ranges2)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_intersection_ranges_lists(self):
        ranges1 = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        ranges2 = [(0, 0.1), (1.0, 1.4), (1.5, 1.6), (1.9, 3.4), (4.3, 4.6)]
        expected_ranges = [(1.1, 1.4), (1.5, 1.6), (1.9, 2.2), (3.3, 3.4), (4.3, 4.4)]
        dst_ranges = intersection_ranges_lists(ranges1, ranges2)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_get_ranges_sum(self):
        ranges = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        expected_sum = 3.3
        dst_sum = get_ranges_sum(ranges)
        self.assertTrue(math.isclose(dst_sum, expected_sum))


if __name__ == '__main__':
    unittest.main()
