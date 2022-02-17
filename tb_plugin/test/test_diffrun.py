import os
import unittest

import pytest
from torch_tb_profiler.profiler.data import RunProfileData
from torch_tb_profiler.profiler.diffrun import (compare_op_tree, diff_summary,
                                                print_node, print_ops)
from torch_tb_profiler.utils import timing


def load_profile(worker, span, path):
    return RunProfileData.parse(worker, span, path, '.')


class TestDiffRun(unittest.TestCase):

    @pytest.mark.skipif(not (os.path.isfile(os.path.expanduser('~/profile_result/worker0.pt.trace.json')) and
                        os.path.isfile(os.path.expanduser('~/profile_result/worker1.pt.trace.json'))),
                        reason="file doesn't exist")
    def test_happy_path(self):
        # path1 = os.path.expanduser('~/profile_result/worker0.pt.trace.json')
        path1 = '/home/mike/git/kineto/tb_plugin/examples/result/datapipe0.1638760942588.pt.trace.json'
        profile1 = load_profile('worker0', 1, path1)
        roots = list(profile1.tid2tree.values())
        root = roots[0]

        # path2 = os.path.expanduser('~/profile_result/worker1.pt.trace.json')
        path2 = '/home/mike/git/kineto/tb_plugin/examples/result/datapipe0.1638835897553.pt.trace.json'
        profile2 = load_profile('worker0', 1, path2)
        roots1 = list(profile2.tid2tree.values())
        root1 = roots1[0]

        with timing('Compare operator tree', True):
            node = compare_op_tree(root, root1)

        print_ops(node.children[4].left, prefix='    ')
        print('========================================================')
        print_ops(node.children[4].right)

        print('*********************** summary *************************')
        with timing('Diff summary', True):
            stats = diff_summary(node)

        # result = stats.flatten_diff_tree()
        # path = '0-1-1'
        # json_data = result[path].get_diff_node_summary(path)
        print_node(stats, 0, 0)


if __name__ == '__main__':
    unittest.main()
