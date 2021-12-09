import os
import unittest

import pytest
from torch_tb_profiler.profiler.data import RunProfileData
from torch_tb_profiler.profiler.diffrun.tree import compare_op_tree, print_node


def load_profile(worker, span, path):
    return RunProfileData.parse(worker, span, path, '.')


class TestDiffRun(unittest.TestCase):

    @pytest.mark.skipif(not (os.path.isfile(os.path.expanduser("~/profile_result/worker0.pt.trace.json")) and
                        os.path.isfile(os.path.expanduser("~/profile_result/worker1.pt.trace.json"))),
                        reason="file doesn't exist")
    def test_happy_path(self):
        path1 = os.path.expanduser('~/profile_result/worker0.pt.trace.json')
        profile1, _ = load_profile('worker0', 1, path1)
        roots = list(profile1.tid2tree.values())
        root = roots[0]

        path2 = os.path.expanduser('~/profile_result/worker1.pt.trace.json')
        profile2, _ = load_profile('worker0', 1, path2)
        roots1 = list(profile2.tid2tree.values())
        root1 = roots1[0]

        node = compare_op_tree(root, root1)
        print_node(node, 0, 0)


if __name__ == '__main__':
    unittest.main()
