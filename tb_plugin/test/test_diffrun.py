import os
import pytest
import unittest

import torch_tb_profiler.profiler.diffrun.node as diffnode
from torch_tb_profiler.profiler.data import RunProfileData


def load_profile(worker, span, path):
    return RunProfileData.parse(worker, span, path)


class TestDiffRun(unittest.TestCase):

    @pytest.mark.skipif(not (os.path.isfile(os.path.expanduser("~/profile_result/worker0.pt.trace.json")) and
                        os.path.isfile(os.path.expanduser("~/profile_result/worker1.pt.trace.json"))),
                        reason="file doesn't exist")
    def test_happy_path(self):
        path1 = os.path.expanduser('~/profile_result/worker0.pt.trace.json')
        profile1, _ = load_profile('worker0', 1, path1)
        roots = list(profile1.tid2tree.values())
        root = roots[0]
        # print(profile1.has_kernel)
        # roots = list(profile1.tid2tree.values())
        # root = roots[0]
        # print(root.name, ": ", type(root))
        # for child in root.children:
        #     if child.name in ('aten::zeros', 'aten::empty', 'aten::zero_'):
        #         # ignore the zeros workaround operators
        #         continue

        #     print("    ", child.name, ": ", type(child))
        #     # print ProfileSteps
        #     for step_child in child.children:
        #         if step_child.name in ('aten::zeros', 'aten::empty', 'aten::zero_'):
        #             # ignore the zeros workaround operators
        #             continue

        #         print("        ", step_child.name, ": ", type(step_child))
        path2 = os.path.expanduser('~/profile_result/worker1.pt.trace.json')
        profile2, _ = load_profile('worker0', 1, path2)
        roots1 = list(profile2.tid2tree.values())
        root1 = roots1[0]

        node = diffnode.create_diff_tree(root, root1)
        print()
        node.print("")


if __name__ == '__main__':
    unittest.main()
