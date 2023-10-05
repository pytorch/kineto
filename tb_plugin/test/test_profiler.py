import gzip
import json
import os
import unittest

from torch_tb_profiler.profiler.data import (DistributedRunProfileData,
                                             RunProfileData)
from torch_tb_profiler.profiler.loader import RunLoader
from torch_tb_profiler.profiler.overall_parser import ProfileRole
from torch_tb_profiler.profiler.gpu_metrics_parser import GPUMetricsParser
from torch_tb_profiler.run import RunProfile

SCHEMA_VERSION = 1
WORKER_NAME = 'worker0'


def parse_json_trace(json_content, worker_name=WORKER_NAME) -> RunProfileData:
    trace_json = json.loads(json_content)
    trace_json = {'schemaVersion': 1, 'traceEvents': trace_json}
    return RunProfileData.from_json(worker_name, 0, trace_json)


'''
All the events in json string are only simulation, not actual generated events.
We removed the data fields that not used by current version of our profiler,
for easy to check correctness and shorter in length.
We even renamed the data values such as kernel name or 'ts', to simplify the string.
'''


class TestProfiler(unittest.TestCase):
    #  A test case including all 7 event categories.
    def test_all_categories(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 180,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::nll_loss_backward", "pid": 13721, "tid": "456",
                "ts": 340, "dur": 70,
                "args": {"Input Dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 50, "dur": 400,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Memcpy",
                "name": "Memcpy HtoD (Pageable -> Device)", "pid": 0, "tid": "stream 7",
                "ts": 405, "dur": 10,
                "args": {"stream": 7, "correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemcpyAsync", "pid": 13721, "tid": "456",
                "ts": 360, "dur": 20,
                "args": {"correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Memset",
                "name": "Memset (Device)", "pid": 0, "tid": "stream 7",
                "ts": 420, "dur": 5,
                "args": {"stream": 7, "correlation": 40344, "external id": 4}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemsetAsync", "pid": 13721, "tid": "456",
                "ts": 390, "dur": 10,
                "args": {"correlation": 40344, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 430, "dur": 15,
                "args": {"correlation": 40348, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 405, "dur": 5,
                "args": {"correlation": 40348, "external id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertTrue(profile.has_runtime)
        self.assertTrue(profile.has_kernel)
        self.assertTrue(profile.has_memcpy_or_memset)
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 15)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 10)
        self.assertEqual(step.costs[ProfileRole.Memset], 5)
        self.assertEqual(step.costs[ProfileRole.Runtime], 30)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 180)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 35)
        self.assertEqual(step.costs[ProfileRole.Other], 125)

        self.assertEqual(len(profile.op_list_groupby_name), 2)
        self.assertEqual(len(profile.op_list_groupby_name_input), 2)

        def test_op_list(op_list):
            op_count = 0
            for op_agg in op_list:
                if op_agg.name == 'aten::to':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape,
                                     '[[2, 8, 5], [], [], [], [], [], [], []]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 0)
                    self.assertEqual(op_agg.self_host_duration, 60)
                    self.assertEqual(op_agg.self_device_duration, 0)
                if op_agg.name == 'aten::nll_loss_backward':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape,
                                     '[[], [32, 1000], [32], [], [], [], []]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 70)
                    self.assertEqual(op_agg.device_duration, 30)
                    self.assertEqual(
                        op_agg.self_host_duration, 70 - 20 - 10 - 5)
                    self.assertEqual(op_agg.self_device_duration, 30)
            self.assertEqual(op_count, 2)

        test_op_list(profile.op_list_groupby_name)
        test_op_list(profile.op_list_groupby_name_input)

        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)
        self.assertEqual(profile.kernel_stat.shape[0], 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].name,
                         'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>')
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].op_name, 'aten::nll_loss_backward')
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].total_duration, 15)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].min_duration, 15)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].max_duration, 15)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]['sum'], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]['mean'], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]['min'], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]['max'], 15)

    # Test using external_id to build relationship between Operator and Runtime.
    # Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
    # Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
    # just use interval containing relationship can't tell it is child or brother of the OperatorNode.
    def test_external_id(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 70,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 130, "dur": 5,
                "args": {"correlation": 334, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 0,
                "args": {"correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 130, "dur": 6,
                "args": {"correlation": 335, "external id": 2, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 0,
                "args": {"correlation": 335, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 130, "dur": 7,
                "args": {"correlation": 336, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 190, "dur": 0,
                "args": {"correlation": 336, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 130, "dur": 8,
                "args": {"correlation": 337, "external id": 2, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 190, "dur": 0,
                "args": {"correlation": 337, "external id": 2}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        op_count = 0
        for op_agg in profile.op_list_groupby_name:
            if op_agg.name == 'aten::mat_mul':
                op_count += 1
                self.assertEqual(op_agg.device_duration, 5 + 6 + 7 + 8)
                self.assertEqual(op_agg.self_device_duration, 6 + 8)
            if op_agg.name == 'aten::mm':
                op_count += 1
                self.assertEqual(op_agg.device_duration, 5 + 7)
                self.assertEqual(op_agg.self_device_duration, 5 + 7)
        self.assertEqual(op_count, 2)

    # Test operator's father-child relationship when they have same start time or end time.
    def test_operator_relation(self):
        # 2 events with same start time.
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 70,
                "args": {"Input Dims": [], "External id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        op_count = 0
        for op_agg in profile.op_list_groupby_name:
            if op_agg.name == 'aten::mat_mul':
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 100 - 70)
            if op_agg.name == 'aten::mm':
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 70)
        self.assertEqual(op_count, 2)

        # 2 events with same end time.
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 130, "dur": 70,
                "args": {"Input Dims": [], "External id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        op_count = 0
        for op_agg in profile.op_list_groupby_name:
            if op_agg.name == 'aten::mat_mul':
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 100 - 70)
            if op_agg.name == 'aten::mm':
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 70)
        self.assertEqual(op_count, 2)

    # Test multiple father-child operators with same name.
    # In this case, all the operators except the top operator should be removed,
    # and all runtime/kernels belong to the children operators should be attached to the only kept one.
    # This behavior is to keep consistent with _remove_dup_nodes in torch/autograd/profiler.py.
    def test_remove_dup_nodes(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 110, "dur": 80,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 60,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 130, "dur": 20,
                "args": {"correlation": 335, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void gemmSN_TN_kernel_64addr", "pid": 0, "tid": "stream 7",
                "ts": 220, "dur": 8,
                "args": {"correlation": 335, "external id": 4, "device": 0}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(len(profile.op_list_groupby_name), 1)
        self.assertEqual(
            profile.op_list_groupby_name[0].self_device_duration, 8)

    # Test Runtime with 'external id' 0.
    # This kind of Runtime should not be attached to any operator,
    # and should be included in accumulating device time.
    def test_top_level_runtime(self):
        # This operator is different thread with the runtime.
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 130, "dur": 20,
                "args": {"correlation": 335, "external id": 335}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void gemmSN_TN_kernel_64addr", "pid": 0, "tid": "stream 7",
                "ts": 220, "dur": 8,
                "args": {"correlation": 335, "external id": 335, "device": 0}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(profile.op_list_groupby_name[0].device_duration, 0)
        self.assertEqual(
            profile.op_list_groupby_name[0].self_device_duration, 0)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)

    # Test Runtime directly called in ProfilerStep, not inside any operator.
    def test_runtime_called_by_profilerstep(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 300,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 130, "dur": 20,
                "args": {"correlation": 335, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void gemmSN_TN_kernel_64addr", "pid": 0, "tid": "stream 7",
                "ts": 220, "dur": 8,
                "args": {"correlation": 335, "external id": 2, "device": 0}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 8)
        self.assertEqual(step.costs[ProfileRole.Runtime], 20)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 0)
        self.assertEqual(step.costs[ProfileRole.Other], 300 - 8 - 20)
        # ProfilerStep is not regarded as an operator.
        self.assertEqual(len(profile.op_list_groupby_name), 0)
        self.assertEqual(len(profile.op_list_groupby_name_input), 0)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

    # Test one Runtime lauch more than one Kernels.
    # Sometimes such as running Bert using DataParallel mode(1 process, 2GPUs),
    # one runtime such as cudaLaunchCooperativeKernelMultiDevice could trigger more than one kernel,
    # each Kernel runs at a seperate GPU card.
    def test_runtime_launch_multipe_kernels(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "Broadcast", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 300,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchCooperativeKernelMultiDevice", "pid": 13721, "tid": "456",
                "ts": 130, "dur": 20,
                "args": {"correlation": 335, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "ncclBroadcastRingLLKernel_copy_i8(ncclColl)", "pid": 0, "tid": "stream 13",
                "ts": 160, "dur": 120318,
                "args": {"device": 0, "context": 1, "stream": 13,
                        "correlation": 335, "external id": 2, "device": 0}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "ncclBroadcastRingLLKernel_copy_i8(ncclColl)", "pid": 0, "tid": "stream 22",
                "ts": 170, "dur": 132800,
                "args": {"device": 0, "context": 2, "stream": 22,
                        "correlation": 335, "external id": 2}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(
            profile.op_list_groupby_name[0].device_duration, 120318 + 132800)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 2)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

    # Test when there is no ProfilerStep#.
    def test_no_profilerstep(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::nll_loss_backward", "pid": 13721, "tid": "456",
                "ts": 300, "dur": 70,
                "args": {"Input Dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 320, "dur": 100,
                "args": {"correlation": 40348, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 310, "dur": 20,
                "args": {"correlation": 40348, "external id": 4}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertTrue(profile.has_runtime)
        self.assertTrue(profile.has_kernel)
        self.assertTrue(not profile.has_memcpy_or_memset)
        self.assertEqual(len(profile.steps_costs), 1)
        step = profile.steps_costs[0]

        self.assertEqual(step.costs[ProfileRole.Kernel], 100)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 320 - 310)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 60 + (310 - 300))
        # If no ProfilerStep, all events will be regarded as a step.
        self.assertEqual(step.costs[ProfileRole.Other], 300 - (100 + 60))
        self.assertEqual(step.costs[ProfileRole.Total], (320 + 100) - 100)
        self.assertEqual(len(profile.op_list_groupby_name), 2)
        self.assertEqual(len(profile.op_list_groupby_name_input), 2)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

        def test_op_list(op_list):
            op_count = 0
            for op_agg in op_list:
                if op_agg.name == 'aten::to':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape,
                                     '[[2, 8, 5], [], [], [], [], [], [], []]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 0)
                    self.assertEqual(op_agg.self_host_duration, 60)
                    self.assertEqual(op_agg.self_device_duration, 0)
                if op_agg.name == 'aten::nll_loss_backward':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape,
                                     '[[], [32, 1000], [32], [], [], [], []]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 70)
                    self.assertEqual(op_agg.device_duration, 100)
                    self.assertEqual(op_agg.self_host_duration, 70 - 20)
                    self.assertEqual(op_agg.self_device_duration, 100)
            self.assertEqual(op_count, 2)

        test_op_list(profile.op_list_groupby_name)
        test_op_list(profile.op_list_groupby_name_input)

        self.assertEqual(profile.kernel_list_groupby_name_op[0].name,
                         'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>')
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].op_name, 'aten::nll_loss_backward')
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].total_duration, 100)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].min_duration, 100)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].max_duration, 100)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]['sum'], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]['mean'], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]['min'], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]['max'], 100)

    # 2 steps without overlap with each other.
    def test_multiple_profilersteps_no_overlap(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 200,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
                "ts": 350, "dur": 150,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "123",
                "ts": 360, "dur": 50,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Memcpy",
                "name": "Memcpy HtoD (Pageable -> Device)", "pid": 0, "tid": "stream 7",
                "ts": 280, "dur": 40,
                "args": {"stream": 7, "correlation": 334, "external id": 2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemcpyAsync", "pid": 13721, "tid": "123",
                "ts": 250, "dur": 5,
                "args": {"correlation": 334, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 410, "dur": 200,
                "args": {"correlation": 40348, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "123",
                "ts": 400, "dur": 5,
                "args": {"correlation": 40348, "external id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertTrue(profile.has_runtime)
        self.assertTrue(profile.has_kernel)
        self.assertTrue(profile.has_memcpy_or_memset)
        self.assertEqual(len(profile.steps_costs), 2)
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 0)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 40)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 60 - 5)
        self.assertEqual(step.costs[ProfileRole.Other], 200 - 60 - 20)
        # Device side takes effect.
        self.assertEqual(step.costs[ProfileRole.Total], 320 - 100)
        step = profile.steps_costs[1]
        self.assertEqual(step.costs[ProfileRole.Kernel], 200)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 50 - 5)
        self.assertEqual(step.costs[ProfileRole.Other], 360 - 350)
        # Device side takes effect.
        self.assertEqual(step.costs[ProfileRole.Total], 610 - 350)
        self.assertEqual(
            profile.avg_costs.costs[ProfileRole.Total], ((320 - 100) + (610 - 350)) / 2)

        self.assertEqual(len(profile.op_list_groupby_name), 2)
        self.assertEqual(len(profile.op_list_groupby_name_input), 2)

        def test_op_list(op_list):
            op_count = 0
            for op_agg in op_list:
                if op_agg.name == 'aten::to':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape,
                                     '[[2, 8, 5], [], [], [], [], [], [], []]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 40)
                    self.assertEqual(op_agg.self_host_duration, 60 - 5)
                    self.assertEqual(op_agg.self_device_duration, 40)
                if op_agg.name == 'aten::mm':
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, '[]')
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 50)
                    self.assertEqual(op_agg.device_duration, 200)
                    self.assertEqual(op_agg.self_host_duration, 50 - 5)
                    self.assertEqual(op_agg.self_device_duration, 200)
            self.assertEqual(op_count, 2)

        test_op_list(profile.op_list_groupby_name)
        test_op_list(profile.op_list_groupby_name_input)

        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)
        self.assertEqual(profile.kernel_stat.shape[0], 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].name,
                         'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>')
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].op_name, 'aten::mm')
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].total_duration, 200)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].min_duration, 200)
        self.assertEqual(
            profile.kernel_list_groupby_name_op[0].max_duration, 200)
        self.assertEqual(profile.kernel_stat.iloc[0]['count'], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]['sum'], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]['mean'], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]['min'], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]['max'], 200)

    # Test self time and total time on operator with nested operator.
    def test_self_time(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 40,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 155, "dur": 20,
                "args": {"correlation": 334, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 150, "dur": 10,
                "args": {"correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 210, "dur": 16,
                "args": {"correlation": 335, "external id": 2, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 170, "dur": 25,
                "args": {"correlation": 335, "external id": 2}
            }]
        """
        profile = parse_json_trace(json_content)

        op_count = 0
        for op_agg in profile.op_list_groupby_name:
            if op_agg.name == 'aten::mat_mul':
                op_count += 1
                self.assertEqual(op_agg.host_duration, 100)
                self.assertEqual(op_agg.device_duration, 20 + 16)
                self.assertEqual(op_agg.self_host_duration, 100 - 40 - 25)
                self.assertEqual(op_agg.self_device_duration, 16)
            if op_agg.name == 'aten::mm':
                op_count += 1
                self.assertEqual(op_agg.host_duration, 40)
                self.assertEqual(op_agg.device_duration, 20)
                self.assertEqual(op_agg.self_host_duration, 30)
                self.assertEqual(op_agg.self_device_duration, 20)
        self.assertEqual(op_count, 2)

    # 2 steps with overlap with each other.
    def test_multiple_profilersteps_with_overlap(self):
        # The kernel with 'correlation' as 123 is launched by previous step,
        # its end time is bigger than 'ProfilerStep#1''s start time,
        # so it is regarded as beginning of 'ProfilerStep#1'.
        # The memcpy with 'correlation' as 334 is launched by 'ProfilerStep#1',
        # its end time is bigger than 'ProfilerStep#2''s start time,
        # so it is regarded as beginning of 'ProfilerStep#2'.
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 200,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
                "ts": 350, "dur": 150,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "123",
                "ts": 360, "dur": 50,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 150, "dur": 90,
                "args": {"correlation": 123, "external id": 123, "device": 0}
            },
            {
                "ph": "X", "cat": "Memcpy",
                "name": "Memcpy HtoD (Pageable -> Device)", "pid": 0, "tid": "stream 7",
                "ts": 280, "dur": 100,
                "args": {"stream": 7, "correlation": 334, "external id": 2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemcpyAsync", "pid": 13721, "tid": "123",
                "ts": 250, "dur": 5,
                "args": {"correlation": 334, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 410, "dur": 200,
                "args": {"correlation": 40348, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "123",
                "ts": 400, "dur": 5,
                "args": {"correlation": 40348, "external id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertTrue(profile.has_runtime)
        self.assertTrue(profile.has_kernel)
        self.assertTrue(profile.has_memcpy_or_memset)
        self.assertEqual(len(profile.steps_costs), 2)
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 0)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 100)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp],
                         (200 + 60) - (150 + 90) - 5)
        self.assertEqual(step.costs[ProfileRole.Other], 280 - (200 + 60))
        # Device side takes effect.
        self.assertEqual(step.costs[ProfileRole.Total],
                         (280 + 100) - (150 + 90))
        step = profile.steps_costs[1]
        self.assertEqual(step.costs[ProfileRole.Kernel], 200)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp],
                         (280 + 100) - 360 + (410 - 405))
        self.assertEqual(step.costs[ProfileRole.Other], 0)
        # Device side takes effect.
        self.assertEqual(step.costs[ProfileRole.Total], 610 - (280 + 100))

    # Test whether step time is calculated correctly when the last 2 steps have no kernels launched.
    def test_last_steps_no_kernel(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 200,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 120, "dur": 10,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
                "ts": 300, "dur": 100,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#3", "pid": 13721, "tid": "123",
                "ts": 400, "dur": 50,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 90, "dur": 20,
                "args": {"correlation": 123, "external id": 123, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemcpyAsync", "pid": 13721, "tid": "123",
                "ts": 125, "dur": 5,
                "args": {"correlation": 334, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 150, "dur": 180,
                "args": {"correlation": 334, "external id": 2, "device": 0}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        # The last 2 steps without kernels are removed from overall view.
        self.assertEqual(len(profile.steps_costs), 1)
        step = profile.steps_costs[0]
        self.assertEqual(
            step.costs[ProfileRole.Total], (150 + 180) - (90 + 20))

    def test_pure_cpu(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 200,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 120, "dur": 10,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
                "ts": 300, "dur": 100,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "123",
                "ts": 350, "dur": 40,
                "args": {"Input Dims": [], "External id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertEqual(len(profile.steps_costs), 2)
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 0)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 0)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 10)
        self.assertEqual(step.costs[ProfileRole.Other], 200 - 10)
        self.assertEqual(step.costs[ProfileRole.Total], 200)
        step = profile.steps_costs[1]
        self.assertEqual(step.costs[ProfileRole.Kernel], 0)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 0)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 40)
        self.assertEqual(step.costs[ProfileRole.Other], 100 - 40)
        self.assertEqual(step.costs[ProfileRole.Total], 100)

    # Test GPU utilization, est. SM efficiency, and occupancy.
    def test_gpu_utilization(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 70,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 1, "tid": "stream 7",
                "ts": 130, "dur": 10,
                "args": {"correlation": 334, "external id": 4, "device": 1,
                        "blocks per SM": 0.5, "est. achieved occupancy %": 0.6}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 0,
                "args": {"correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void gemmSN_TN_kernel_64addr", "pid": 1, "tid": "stream 8",
                "ts": 135, "dur": 15,
                "args": {"correlation": 335, "external id": 2, "device": 1,
                        "blocks per SM": 0.6, "est. achieved occupancy %": 0.1}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void gemmSN_TN_kernel_64addr", "pid": 1, "tid": "stream 8",
                "ts": 150, "dur": 0,
                "args": {"correlation": 335, "external id": 2, "device": 1,
                        "blocks per SM": 0.3, "est. achieved occupancy %": 0.2}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 0,
                "args": {"correlation": 335, "external id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 1, "tid": "stream 7",
                "ts": 145, "dur": 25,
                "args": {"correlation": 336, "external id": 4, "device": 1,
                        "blocks per SM": 0.3, "est. achieved occupancy %": 1.0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 125, "dur": 3,
                "args": {"correlation": 336, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 1, "tid": "stream 7",
                "ts": 200, "dur": 20,
                "args": {"correlation": 337, "external id": 2, "device": 1,
                        "blocks per SM": 10.5, "est. achieved occupancy %": 0.3}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 195, "dur": 1,
                "args": {"correlation": 337, "external id": 2}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertEqual(len(profile.gpu_metrics_parser.gpu_ids), 1)
        self.assertAlmostEqual(profile.gpu_metrics_parser.gpu_utilization[1], (40 + 20) / 120)
        self.assertAlmostEqual(profile.gpu_metrics_parser.avg_approximated_sm_efficiency_per_device[1],
                               (0.5 * (135 - 130)
                                + 1.0 * (140 - 135)
                                + 0.6 * (145 - 140)
                                + 0.9 * (150 - 145)
                                + 0.3 * (170 - 150)
                                + 1.0 * (220 - 200)) / (220 - 100))
        self.assertAlmostEqual(profile.gpu_metrics_parser.avg_occupancy_per_device[1],
                               (0.6 * 10 + 0.1 * 15 + 1.0 * 25 + 0.3 * 20) / (10 + 15 + 25 + 20))

        gpu_util_expected = [(100, 0), (110, 0), (120, 0), (130, 1.0), (140, 1.0), (150, 1.0), (160, 1.0),
                             (170, 0), (180, 0), (190, 0), (200, 1.0), (210, 1.0), (220, 0)]
        for gpu_id in profile.gpu_metrics_parser.gpu_ids:
            buckets = profile.gpu_metrics_parser.gpu_util_buckets[gpu_id]
            gpu_util_id = 0
            for b in buckets:
                self.assertEqual(b[0], gpu_util_expected[gpu_util_id][0])
                self.assertAlmostEqual(b[1], gpu_util_expected[gpu_util_id][1])
                gpu_util_id += 1
            self.assertEqual(gpu_util_id, len(gpu_util_expected))

        sm_efficiency_expected = [(130, 0.5), (135, 0), (135, 1.0), (140, 0), (140, 0.6), (145, 0), (145, 0.9),
                                  (150, 0), (150, 0.3), (170, 0), (170, 0), (200, 0), (200, 1.0), (220, 0)]
        for gpu_id in profile.gpu_metrics_parser.gpu_ids:
            ranges = profile.gpu_metrics_parser.approximated_sm_efficiency_ranges[gpu_id]
            sm_efficiency_id = 0
            for r in ranges:
                self.assertEqual(
                    r[0], sm_efficiency_expected[sm_efficiency_id][0])
                self.assertAlmostEqual(
                    r[2], sm_efficiency_expected[sm_efficiency_id][1])
                sm_efficiency_id += 1
                self.assertEqual(
                    r[1], sm_efficiency_expected[sm_efficiency_id][0])
                self.assertAlmostEqual(
                    0, sm_efficiency_expected[sm_efficiency_id][1])
                sm_efficiency_id += 1
            self.assertEqual(sm_efficiency_id, len(sm_efficiency_expected))

        count = 0
        for agg_by_op in profile.kernel_list_groupby_name_op:
            if agg_by_op.name == 'void gemmSN_TN_kernel_64addr' and agg_by_op.op_name == 'aten::mat_mul':
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, 0.6)
                self.assertAlmostEqual(agg_by_op.avg_occupancy, 0.1)
                count += 1
            if agg_by_op.name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>' and \
                    agg_by_op.op_name == 'aten::mm':
                self.assertAlmostEqual(
                    agg_by_op.avg_blocks_per_sm, (0.5 * 10 + 0.3 * 25) / (10 + 25))
                self.assertAlmostEqual(
                    agg_by_op.avg_occupancy, (0.6 * 10 + 1.0 * 25) / (10 + 25))
                count += 1
            if agg_by_op.name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>' and \
                    agg_by_op.op_name == 'aten::mat_mul':
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, 10.5)
                self.assertAlmostEqual(agg_by_op.avg_occupancy, 0.3)
                count += 1
        self.assertEqual(count, 3)

        count = 0
        for _id, (name, row) in enumerate(profile.kernel_stat.iterrows()):
            # The kernel with zero 'dur' should be ignored.
            if name == 'void gemmSN_TN_kernel_64addr':
                self.assertAlmostEqual(row['blocks_per_sm'], 0.6)
                self.assertAlmostEqual(row['occupancy'], 0.1)
                count += 1
            if name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>':
                self.assertAlmostEqual(
                    row['blocks_per_sm'], (0.5 * 10 + 0.3 * 25 + 10.5 * 20) / (10 + 25 + 20))
                self.assertAlmostEqual(
                    row['occupancy'], (0.6 * 10 + 1.0 * 25 + 0.3 * 20) / (10 + 25 + 20))
                count += 1
        self.assertEqual(count, 2)

    # Test GPU utilization 3 metrics works fine if kernel out of ProfilerStep.
    def test_gpu_utilization_kernel_out_of_step(self):
        json_content = """
            [{
                "ph": "X", "cat": "Operator",
                "name": "aten::mat_mul", "pid": 13721, "tid": "456",
                "ts": 10, "dur": 10,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 70,
                "args": {"Input Dims": [], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 220, "dur": 20,
                "args": {"Input Dims": [], "External id": 4}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "456",
                "ts": 100, "dur": 100,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 1, "tid": "stream 7",
                "ts": 60, "dur": 20,
                "args": {"correlation": 334, "external id": 1, "device": 1,
                        "blocks per SM": 0.5, "est. achieved occupancy %": 0.6}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 15, "dur": 5,
                "args": {"correlation": 334, "external id": 1}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 1, "tid": "stream 7",
                "ts": 240, "dur": 25,
                "args": {"correlation": 337, "external id": 4, "device": 1,
                        "blocks per SM": 10.5, "est. achieved occupancy %": 0.3}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 230, "dur": 10,
                "args": {"correlation": 337, "external id": 4}
            }]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        self.assertEqual(len(profile.gpu_metrics_parser.gpu_ids), 1)
        self.assertAlmostEqual(profile.gpu_metrics_parser.gpu_utilization[1], 0.0)
        self.assertTrue(profile.gpu_metrics_parser.avg_approximated_sm_efficiency_per_device[1] is None)
        self.assertTrue(profile.gpu_metrics_parser.avg_occupancy_per_device[1] is None)
        self.assertTrue(profile.gpu_metrics_parser.blocks_per_sm_count[1] > 0)
        self.assertTrue(profile.gpu_metrics_parser.occupancy_count[1] > 0)

        count = 0
        for agg_by_op in profile.kernel_list_groupby_name_op:
            if agg_by_op.name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>' \
                    and agg_by_op.op_name == 'aten::mat_mul':
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, 0.5)
                self.assertAlmostEqual(agg_by_op.avg_occupancy, 0.6)
                count += 1
            if agg_by_op.name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>' and \
                    agg_by_op.op_name == 'aten::mm':
                self.assertAlmostEqual(
                    agg_by_op.avg_blocks_per_sm, 10.5)
                self.assertAlmostEqual(
                    agg_by_op.avg_occupancy, 0.3)
                count += 1
        self.assertEqual(count, 2)

        count = 0
        for _id, (name, row) in enumerate(profile.kernel_stat.iterrows()):
            # The kernel with zero 'dur' should be ignored.
            if name == 'void cunn_ClassNLLCriterion_updateGradInput_kernel<float>':
                self.assertAlmostEqual(row['blocks_per_sm'], (20 * 0.5 + 25 * 10.5) / (20 + 25))
                self.assertAlmostEqual(row['occupancy'], (20 * 0.6 + 25 * 0.3) / (20 + 25))
                count += 1
        self.assertEqual(count, 1)

    def test_dump_gpu_metrics(self):
        profile = RunProfile('test_dump_gpu_metrics', None)
        # Faked data for easy to see in UI. Real data values are 1/100 of these.
        gpu_util_buckets = [[(1621401187223005, 0.0), (1621401187224005, 0.0),
                            (1621401187225005, 0.6), (1621401187226005, 0.5),
                            (1621401187227005, 0.6), (1621401187228005, 0.2),
                            (1621401187229005, 0.6), (1621401187230005, 0.1),
                            (1621401187231005, 0.5), (1621401187232005, 0.2),
                            (1621401187233005, 0.3), (1621401187234005, 0.4),
                            (1621401187235005, 0.4219409282700422),
                            (1621401187236901, 0)]]
        # Faked data for easy to see in UI. Real data values are 1/10 of these.
        approximated_sm_efficiency_ranges = \
            [[(1621401187225275, 1621401187225278, 0.25), (1621401187225530, 1621401187225532, 0.125),
              (1621401187225820, 1621401187225821, 0.125), (1621401187226325, 1621401187226327, 0.25),
              (1621401187226575, 1621401187226577, 0.125), (1621401187226912, 1621401187226913, 0.125),
              (1621401187227092, 1621401187227094, 0.125), (1621401187227619, 1621401187227620, 0.125),
              (1621401187227745, 1621401187227746, 0.125), (1621401187227859, 1621401187227860, 0.125),
              (1621401187227973, 1621401187227974, 0.125), (1621401187228279, 1621401187228280, 0.125),
              (1621401187228962, 1621401187228963, 0.125), (1621401187229153, 1621401187229155, 0.125),
              (1621401187229711, 1621401187229715, 0.125), (1621401187230162, 1621401187230163, 0.125),
              (1621401187231100, 1621401187231103, 0.125), (1621401187231692, 1621401187231694, 0.5),
              (1621401187232603, 1621401187232604, 0.125), (1621401187232921, 1621401187232922, 0.125),
              (1621401187233342, 1621401187233343, 0.125), (1621401187233770, 1621401187233772, 0.125),
              (1621401187234156, 1621401187234159, 0.125), (1621401187234445, 1621401187234446, 0.125),
              (1621401187235025, 1621401187235028, 0.125), (1621401187235555, 1621401187235556, 0.125),
              (1621401187236158, 1621401187236159, 0.125), (1621401187236278, 1621401187236279, 0.125),
              (1621401187236390, 1621401187236391, 0.125), (1621401187236501, 1621401187236502, 0.125)]]

        basedir = os.path.dirname(os.path.realpath(__file__))
        trace_json_flat_path = os.path.join(basedir, 'gpu_metrics_input.json')
        gpu_metrics_parser = GPUMetricsParser()
        gpu_metrics_parser.gpu_util_buckets = gpu_util_buckets
        gpu_metrics_parser.approximated_sm_efficiency_ranges = approximated_sm_efficiency_ranges
        profile.gpu_metrics = gpu_metrics_parser.get_gpu_metrics()
        with open(trace_json_flat_path, 'rb') as file:
            raw_data = file.read()
        data_with_gpu_metrics_compressed = profile.append_gpu_metrics(raw_data)
        data_with_gpu_metrics_flat = gzip.decompress(
            data_with_gpu_metrics_compressed)

        trace_json_expected_path = os.path.join(basedir, 'gpu_metrics_expected.json')
        with open(trace_json_expected_path, 'rb') as file:
            data_expected = file.read()

        # Parse to json in order to ignore text format difference.
        data_with_gpu_metrics_json = json.loads(
            data_with_gpu_metrics_flat.decode('utf8'))
        data_expected_json = json.loads(data_expected.decode('utf8'))
        data_with_gpu_metrics_str = json.dumps(
            data_with_gpu_metrics_json, sort_keys=True)
        data_expected_str = json.dumps(data_expected_json, sort_keys=True)

        self.assertEqual(data_with_gpu_metrics_str, data_expected_str)

        try:
            _ = json.loads(data_with_gpu_metrics_flat.decode('utf8'))
        except Exception:
            self.assertTrue(
                False, 'The string fails to be parsed by json after appending gpu metrics.')

    def test_memory_view(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 10, "dur": 10,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", "pid": 13721, "tid": "123",
                "ts": 100, "dur": 180,
                "args": {"Input Dims": [], "External id": 2}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::nll_loss_backward", "pid": 13721, "tid": "123",
                "ts": 340, "dur": 70,
                "args": {"Input Dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
                "ts": 50, "dur": 400,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
                "ts": 500, "dur": 500,
                "args": {"Input Dims": [], "External id": 1}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 510, "dur": 150,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::copy_", "pid": 13721, "tid": "123",
                "ts": 520, "dur": 100,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },

            {
                "ph": "X", "cat": "Operator",
                "name": "aten::liner", "pid": 13721, "tid": "123",
                "ts": 700, "dur": 100,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::t", "pid": 13721, "tid": "123",
                "ts": 705, "dur": 40,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::transpose", "pid": 13721, "tid": "123",
                "ts": 710, "dur": 30,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::tranas_stride", "pid": 13721, "tid": "123",
                "ts": 720, "dur": 10,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::addmm", "pid": 13721, "tid": "123",
                "ts": 750, "dur": 40,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::to", "pid": 13721, "tid": "123",
                "ts": 900, "dur": 100,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Memcpy",
                "name": "Memcpy HtoD (Pageable -> Device)", "pid": 0, "tid": "stream 7",
                "ts": 405, "dur": 10,
                "args": {"stream": 7, "correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemcpyAsync", "pid": 13721, "tid": "456",
                "ts": 360, "dur": 20,
                "args": {"correlation": 334, "external id": 4}
            },
            {
                "ph": "X", "cat": "Memset",
                "name": "Memset (Device)", "pid": 0, "tid": "stream 7",
                "ts": 420, "dur": 5,
                "args": {"stream": 7, "correlation": 40344, "external id": 4}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaMemsetAsync", "pid": 13721, "tid": "456",
                "ts": 390, "dur": 10,
                "args": {"correlation": 40344, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 430, "dur": 15,
                "args": {"correlation": 40348, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 405, "dur": 5,
                "args": {"correlation": 40348, "external id": 4}
            },


            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 90,
                "args": {
                "Device Type": 0, "Device Id": -1, "Addr": 90, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 150,
                "args": {
                "Device Type": 0, "Device Id": -1, "Addr": 150, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 200,
                "args": {
                "Device Type": 0, "Device Id": -1, "Addr": 200, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 210,
                "args": {
                "Device Type": 1, "Device Id": 0, "Addr": 210, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 265,
                "args": {
                "Device Type": 1, "Device Id": 0, "Addr": 265, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 300,
                "args": {
                "Device Type": 1, "Device Id": 0, "Addr": 300, "Bytes": 4
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 350,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 350, "Bytes": 10
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 360,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 350, "Bytes": -10
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 450,
                "args": {
                    "Device Type": 0, "Device Id": -1, "Addr": 450, "Bytes": 1000000
                }
            },

            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 515,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 515, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 520,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 520, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 600,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 520, "Bytes": -100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 690,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 690, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 701,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 701, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 796,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 515, "Bytes": -100
                }
            },

            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 708,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 708, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 742,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 708, "Bytes": -100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 715,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 715, "Bytes": 50
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 735,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 715, "Bytes": -50
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 725,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 725, "Bytes": 50
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 728,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 725, "Bytes": -50
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 729,
                "args": {
                    "Device Type": 0, "Device Id": -1, "Addr": 729, "Bytes": 50
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 746,
                "args": {
                    "Device Type": 0, "Device Id": -1, "Addr": 746, "Bytes": 100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 747,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 747, "Bytes": 20
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 749,
                "args": {
                    "Device Type": 0, "Device Id": -1, "Addr": 690, "Bytes": -100
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 760,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 760, "Bytes": 30
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 780,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 760, "Bytes": -30
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 795,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 795, "Bytes": 10
                }
            },
            {
                "ph": "i", "s": "t", "name": "[memory]",
                "pid": 13721, "tid": 123,
                "ts": 799,
                "args": {
                    "Device Type": 1, "Device Id": 0, "Addr": 795, "Bytes": -10
                }
            }
        ]
        """
        import logging

        from torch_tb_profiler.utils import get_logger
        logger = get_logger()
        logger.addHandler(logging.StreamHandler())

        profile = parse_json_trace(json_content)
        profile.process()
        memory_stats = profile.memory_snapshot.get_memory_statistics(profile.tid2tree)

        self.assertEqual(len(memory_stats), 2)
        self.assertIn('GPU0', memory_stats)

        # validation
        gpu_expected_data = {
            # self increase size, self allocation size, self allocation count, increase size, allocation size, allocation count, call # noqa: E501
            'aten::to': [104, 104, 2, 104, 204, 3, 4],
            'aten::nll_loss_backward': [0, 10, 1, 0, 10, 1, 1],
            'aten::copy_': [0, 100, 1, 0, 100, 1, 1],
            'aten::addmm': [0, 30, 1, 0, 30, 1, 1],
            'aten::tranas_stride': [0, 50, 1, 0, 50, 1, 1],
            'aten::transpose': [0, 50, 1, 0, 100, 2, 1],
            'aten::t': [0, 100, 1, 0, 200, 3, 1],
            'aten::liner': [20, 130, 3, 20, 360, 7, 1]
        }

        cpu_expected_data = {
            'aten::to': [4, 4, 1, 4, 4, 1, 4],
            'aten::liner': [0, 100, 1, 50, 150, 2, 1],
            'aten::tranas_stride': [50, 50, 1, 50, 50, 1, 1],
            'aten::transpose': [0, 0, 0, 50, 50, 1, 1],
            'aten::t': [0, 0, 0, 50, 50, 1, 1]
        }

        validate_data = [
            (memory_stats['CPU'], cpu_expected_data),
            (memory_stats['GPU0'], gpu_expected_data)
        ]
        for (mem_stat, expected_data) in validate_data:
            for name, values in expected_data.items():
                self.assertEqual(mem_stat[name], values)

    # Test group by 'kernel detail + op name'.
    def test_group_by_kernel_columns(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "op1", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "op2", "pid": 13721, "tid": "456",
                "ts": 340, "dur": 70,
                "args": {"Input Dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 230, "dur": 15,
                "args": {"correlation": 1000, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 16], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 210, "dur": 5,
                "args": {"correlation": 1000, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 250, "dur": 10,
                "args": {"correlation": 1001, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 16], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 215, "dur": 5,
                "args": {"correlation": 1001, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 250, "dur": 13,
                "args": {"correlation": 1002, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 64], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 220, "dur": 5,
                "args": {"correlation": 1002, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 250, "dur": 17,
                "args": {"correlation": 1003, "external id": 4, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 64], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 350, "dur": 5,
                "args": {"correlation": 1003, "external id": 4}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        expected_agg_kernels = [
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': '[16, 1, 1]',
                'block': '[16, 16, 16]',
                'registers per thread': 18,
                'shared memory': 0,
                'calls': 2,
                'total_duration': 15 + 10,
                'avg_duration': (15 + 10) / 2,
                'min_duration': min(15, 10),
                'max_duration': max(15, 10)
            },
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': '[16, 1, 1]',
                'block': '[16, 16, 64]',  # Only changed this.
                'registers per thread': 18,
                'shared memory': 0,
                'calls': 1,
                'total_duration': 13,
                'avg_duration': 13,
                'min_duration': 13,
                'max_duration': 13
            },
            {
                'name': 'kernel1',
                'op_name': 'op2',  # Only changed this.
                'grid': '[16, 1, 1]',
                'block': '[16, 16, 64]',
                'registers per thread': 18,
                'shared memory': 0,
                'calls': 1,
                'total_duration': 17,
                'avg_duration': 17,
                'min_duration': 17,
                'max_duration': 17
            }
        ]
        index = 0
        self.assertEqual(len(profile.kernel_list_groupby_name_op), len(expected_agg_kernels))
        for agg_kernel in profile.kernel_list_groupby_name_op:
            expected_agg_kernel = expected_agg_kernels[index]
            self.assertEqual(agg_kernel.name, expected_agg_kernel['name'])
            self.assertEqual(agg_kernel.op_name, expected_agg_kernel['op_name'])
            self.assertEqual(str(agg_kernel.grid), expected_agg_kernel['grid'])
            self.assertEqual(str(agg_kernel.block), expected_agg_kernel['block'])
            self.assertEqual(agg_kernel.regs_per_thread, expected_agg_kernel['registers per thread'])
            self.assertEqual(agg_kernel.shared_memory, expected_agg_kernel['shared memory'])
            self.assertEqual(agg_kernel.calls, expected_agg_kernel['calls'])
            self.assertEqual(agg_kernel.total_duration, expected_agg_kernel['total_duration'])
            self.assertAlmostEqual(agg_kernel.avg_duration, expected_agg_kernel['avg_duration'])
            self.assertEqual(agg_kernel.min_duration, expected_agg_kernel['min_duration'])
            self.assertEqual(agg_kernel.max_duration, expected_agg_kernel['max_duration'])
            index += 1

    # Test group by 'kernel detail + op name' with invalid input lack of some kernel field
    def test_group_by_kernel_columns_invalid_input(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "op1", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 60,
                "args": {"Input Dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 220, "dur": 1,
                "args": {"correlation": 1000, "external id": 3, "device": 0,
                        "block": [16, 16, 16], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 210, "dur": 5,
                "args": {"correlation": 1000, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 230, "dur": 2,
                "args": {"correlation": 1001, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "registers per thread": 18, "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 220, "dur": 5,
                "args": {"correlation": 1001, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 240, "dur": 3,
                "args": {"correlation": 1002, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 16], "shared memory": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 230, "dur": 5,
                "args": {"correlation": 1002, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 250, "dur": 4,
                "args": {"correlation": 1003, "external id": 3, "device": 0,
                        "grid": [16, 1, 1], "block": [16, 16, 16], "registers per thread": 18}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 240, "dur": 5,
                "args": {"correlation": 1003, "external id": 3}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel1", "pid": 0, "tid": "stream 7",
                "ts": 260, "dur": 5,
                "args": {"correlation": 1004, "external id": 3, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
                "ts": 250, "dur": 5,
                "args": {"correlation": 1004, "external id": 3}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        expected_agg_kernels = [
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': None,
                'block': [16, 16, 16],
                'registers per thread': 18,
                'shared memory': 0,
                'calls': 1,
                'total_duration': 1,
                'avg_duration': 1,
                'min_duration': 1,
                'max_duration': 1
            },
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': [16, 1, 1],
                'block': None,
                'registers per thread': 18,
                'shared memory': 0,
                'calls': 1,
                'total_duration': 2,
                'avg_duration': 2,
                'min_duration': 2,
                'max_duration': 2
            },
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': [16, 1, 1],
                'block': [16, 16, 16],
                'registers per thread': None,
                'shared memory': 0,
                'calls': 1,
                'total_duration': 3,
                'avg_duration': 3,
                'min_duration': 3,
                'max_duration': 3
            },
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': [16, 1, 1],
                'block': [16, 16, 16],
                'registers per thread': 18,
                'shared memory': None,
                'calls': 1,
                'total_duration': 4,
                'avg_duration': 4,
                'min_duration': 4,
                'max_duration': 4
            },
            {
                'name': 'kernel1',
                'op_name': 'op1',
                'grid': None,
                'block': None,
                'registers per thread': None,
                'shared memory': None,
                'calls': 1,
                'total_duration': 5,
                'avg_duration': 5,
                'min_duration': 5,
                'max_duration': 5
            }
        ]
        index = 0
        self.assertEqual(len(profile.kernel_list_groupby_name_op), len(expected_agg_kernels))
        for agg_kernel in profile.kernel_list_groupby_name_op:
            expected_agg_kernel = expected_agg_kernels[index]
            self.assertEqual(agg_kernel.name, expected_agg_kernel['name'])
            self.assertEqual(agg_kernel.op_name, expected_agg_kernel['op_name'])
            self.assertEqual(agg_kernel.grid, expected_agg_kernel['grid'])
            self.assertEqual(agg_kernel.block, expected_agg_kernel['block'])
            self.assertEqual(agg_kernel.regs_per_thread, expected_agg_kernel['registers per thread'])
            print(agg_kernel.name, agg_kernel.grid, agg_kernel.block, agg_kernel.shared_memory)
            self.assertEqual(agg_kernel.shared_memory, expected_agg_kernel['shared memory'])
            self.assertEqual(agg_kernel.calls, expected_agg_kernel['calls'])
            self.assertEqual(agg_kernel.total_duration, expected_agg_kernel['total_duration'])
            self.assertAlmostEqual(agg_kernel.avg_duration, expected_agg_kernel['avg_duration'])
            self.assertEqual(agg_kernel.min_duration, expected_agg_kernel['min_duration'])
            self.assertEqual(agg_kernel.max_duration, expected_agg_kernel['max_duration'])
            index += 1

    # Test tensor core related feature.
    def test_tensor_core(self):
        json_content = """[
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::conv2d", "pid": 13721, "tid": "123",
                "ts": 200, "dur": 100,
                "args": {"Input Dims": [[]], "External id": 3}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "op_no_tc", "pid": 13721, "tid": "123",
                "ts": 205, "dur": 10,
                "args": {"Input Dims": [[]], "External id": 4}
            },
            {
                "ph": "X", "cat": "Operator",
                "name": "aten::cudnn_convolution", "pid": 13721, "tid": "123",
                "ts": 215, "dur": 10,
                "args": {"Input Dims": [[]], "External id": 5}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "kernel_no_tc", "pid": 0, "tid": "stream 7",
                "ts": 210, "dur": 10,
                "args": {"correlation": 1000, "external id": 4, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "123",
                "ts": 205, "dur": 5,
                "args": {"correlation": 1000, "external id": 4}
            },
            {
                "ph": "X", "cat": "Kernel",
                "name": "volta_fp16_s884cudnn_fp16_128x128_ldg8_splitK_relu_f2f_exp_small_nhwc_tn_v1",
                "pid": 0, "tid": "stream 7",
                "ts": 220, "dur": 15,
                "args": {"correlation": 1001, "external id": 5, "device": 0}
            },
            {
                "ph": "X", "cat": "Runtime",
                "name": "cudaLaunchKernel", "pid": 13721, "tid": "123",
                "ts": 215, "dur": 5,
                "args": {"correlation": 1001, "external id": 5}
            }
        ]
        """
        profile = parse_json_trace(json_content)
        profile.process()

        expected_agg_ops = {
            'aten::conv2d': {
                'tc_eligible': True,
                'tc_self_ratio': 0,
                'tc_total_ratio': 15 / (15 + 10)
            },
            'op_no_tc': {
                'tc_eligible': False,
                'tc_self_ratio': 0,
                'tc_total_ratio': 0
            },
            'aten::cudnn_convolution': {
                'tc_eligible': True,
                'tc_self_ratio': 1.0,
                'tc_total_ratio': 1.0
            }
        }
        self.assertEqual(len(profile.op_list_groupby_name), len(expected_agg_ops))
        for agg_op in profile.op_list_groupby_name:
            expected_agg_op = expected_agg_ops[agg_op.name]
            self.assertEqual(agg_op.tc_eligible, expected_agg_op['tc_eligible'])
            self.assertAlmostEqual(agg_op.tc_self_ratio, expected_agg_op['tc_self_ratio'])
            self.assertAlmostEqual(agg_op.tc_total_ratio, expected_agg_op['tc_total_ratio'])

        expected_kernels_groupby_op = {
            'kernel_no_tc': {
                'op_name': 'op_no_tc',
                'tc_used': False,
                'op_tc_eligible': False
            },
            'volta_fp16_s884cudnn_fp16_128x128_ldg8_splitK_relu_f2f_exp_small_nhwc_tn_v1': {
                'op_name': 'aten::cudnn_convolution',
                'tc_used': True,
                'op_tc_eligible': True
            }
        }
        self.assertEqual(len(profile.kernel_list_groupby_name_op), len(expected_kernels_groupby_op))
        for agg_kernel in profile.kernel_list_groupby_name_op:
            expected_agg_kernel = expected_kernels_groupby_op[agg_kernel.name]
            self.assertEqual(agg_kernel.op_name, expected_agg_kernel['op_name'])
            self.assertEqual(agg_kernel.tc_used, expected_agg_kernel['tc_used'])
            self.assertEqual(agg_kernel.op_tc_eligible, expected_agg_kernel['op_tc_eligible'])

        self.assertAlmostEqual(profile.tc_ratio[0], 15 / (15 + 10))
        self.assertAlmostEqual(profile.tc_eligible_ops_kernel_ratio, 15 / (15 + 10))


class TestDistributed(unittest.TestCase):

    def test_distributed_nccl(self):
        json_content0 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "nccl:broadcast", "pid": 23803, "tid": "23803",
            "ts": 0, "dur": 75,
            "args": {"External id": 146, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "ncclKernel_Broadcast_RING_LL_Sum_int8_t(ncclWorkElem)", "pid": 0, "tid": "stream 16",
            "ts": 16, "dur": 16,
            "args": {"device": 0, "correlation": 28506, "external id": 146}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23803",
            "ts": 100, "dur": 20,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "void at::native::vectorized_elementwise_kernel", "pid": 0, "tid": "stream 7",
            "ts": 130, "dur": 161,
            "args": {"device": 0, "correlation": 99765, "external id": 24504}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "nccl:all_reduce", "pid": 23803, "tid": "25166",
            "ts": 160, "dur": 75,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)", "pid": 0, "tid": "stream 16",
            "ts": 162, "dur": 1556,
            "args": {"device": 0, "correlation": 33218, "external id": 2513}
            }
        ]
        """
        json_content1 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "nccl:broadcast", "pid": 23803, "tid": "23803",
            "ts": 0, "dur": 20,
            "args": {"External id": 146, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "ncclKernel_Broadcast_RING_LL_Sum_int8_t(ncclWorkElem)", "pid": 0, "tid": "stream 16",
            "ts": 8, "dur": 31,
            "args": {"device": 0, "correlation": 28506, "external id": 146}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23803",
            "ts": 25, "dur": 20,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "void at::native::vectorized_elementwise_kernel", "pid": 0, "tid": "stream 7",
            "ts": 30, "dur": 161,
            "args": {"device": 0, "correlation": 99765, "external id": 24504}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "nccl:all_reduce", "pid": 23803, "tid": "25166",
            "ts": 160, "dur": 75,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)", "pid": 0, "tid": "stream 16",
            "ts": 562, "dur": 1058,
            "args": {"device": 0, "correlation": 33218, "external id": 2513}
            }
        ]
        """

        profile0 = parse_json_trace(json_content0, 'worker0')
        dist_data0 = DistributedRunProfileData(profile0)
        self.assertTrue(profile0.has_communication)
        self.assertEqual(len(profile0.comm_node_list), 2)
        self.assertEqual(profile0.steps_costs[0].costs, [105, 0, 0, 75, 0, 0, 20, 35, 235])

        profile1 = parse_json_trace(json_content1, 'worker1')
        dist_data1 = DistributedRunProfileData(profile1)
        self.assertTrue(profile1.has_communication)
        self.assertEqual(len(profile1.comm_node_list), 2)
        self.assertEqual(profile1.steps_costs[0].costs[3], 74)

        loader = RunLoader('test_nccl', '', None)
        dist_profile = loader._process_distributed_profiles([dist_data0, dist_data1], 0)
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker0'], [30, 75, 75, 55])
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker1'], [121, 40, 74, 0])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker0'], [1169, 464])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker1'], [1169, 3])
        self.assertEqual(dist_profile.comm_ops['data']['worker0']['rows'],
                         [['nccl:broadcast', 1, 212480, 212480, 75, 75, 36, 36],
                          ['nccl:all_reduce', 1, 8196000, 8196000, 1558, 1558, 1133, 1133]])
        self.assertEqual(dist_profile.comm_ops['data']['worker1']['rows'],
                         [['nccl:broadcast', 1, 212480, 212480, 39, 39, 36, 36],
                          ['nccl:all_reduce', 1, 8196000, 8196000, 1133, 1133, 1133, 1133]])

    def test_distributed_gloo_gpu(self):
        json_content0 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 16, "dur": 38,
            "args": {"External id": 165, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23805",
            "ts": 25, "dur": 36,
            "args": {"External id": 166, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 66, "dur": 18,
            "args": {"External id": 167, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23800",
            "ts": 0, "dur": 20,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "void at::native::vectorized_elementwise_kernel", "pid": 0, "tid": "stream 7",
            "ts": 30, "dur": 101,
            "args": {"device": 0, "correlation": 99765, "external id": 24504}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23805",
            "ts": 110, "dur": 18,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23803",
            "ts": 120, "dur": 36,
            "args": {"External id": 2516, "Input Dims": [[2049000]], "Input type": ["float"]}
            }
        ]
        """
        json_content1 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 20, "dur": 28,
            "args": {"External id": 256, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23805",
            "ts": 28, "dur": 30,
            "args": {"External id": 257, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 77, "dur": 6,
            "args": {"External id": 258, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23800",
            "ts": 0, "dur": 30,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "Kernel",
            "name": "void at::native::vectorized_elementwise_kernel", "pid": 0, "tid": "stream 7",
            "ts": 70, "dur": 70,
            "args": {"device": 0, "correlation": 99765, "external id": 24504}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23805",
            "ts": 88, "dur": 38,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23803",
            "ts": 130, "dur": 16,
            "args": {"External id": 2516, "Input Dims": [[2049000]], "Input type": ["float"]}
            }
        ]
        """

        profile0 = parse_json_trace(json_content0, 'worker0')
        dist_data0 = DistributedRunProfileData(profile0)
        self.assertTrue(profile0.has_communication)
        self.assertEqual(len(profile0.comm_node_list), 5)
        self.assertEqual(profile0.steps_costs[0].costs, [101, 0, 0, 39, 0, 0, 16, 0, 156])

        profile1 = parse_json_trace(json_content1, 'worker1')
        dist_data1 = DistributedRunProfileData(profile1)
        self.assertTrue(profile1.has_communication)
        self.assertEqual(len(profile1.comm_node_list), 5)
        self.assertEqual(profile1.steps_costs[0].costs, [70, 0, 0, 44, 0, 0, 20, 12, 146])

        loader = RunLoader('test_gloo_gpu', '', None)
        dist_profile = loader._process_distributed_profiles([dist_data0, dist_data1], 0)
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker0'], [31, 70, 39, 16])
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker1'], [16, 54, 44, 32])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker0'], [75, 34])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker1'], [78, 20])
        self.assertEqual(dist_profile.comm_ops['data']['worker0']['rows'],
                         [['gloo:broadcast', 3, 637440, 212480, 63, 21, 41, 14],
                          ['gloo:all_reduce', 2, 16392000, 8196000, 46, 23, 34, 17]])
        self.assertEqual(dist_profile.comm_ops['data']['worker1']['rows'],
                         [['gloo:broadcast', 3, 637440, 212480, 44, 15, 44, 15],
                          ['gloo:all_reduce', 2, 16392000, 8196000, 54, 27, 34, 17]])

    def test_distributed_gloo_cpu(self):
        json_content0 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 16, "dur": 38,
            "args": {"External id": 165, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23805",
            "ts": 25, "dur": 36,
            "args": {"External id": 166, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 66, "dur": 18,
            "args": {"External id": 167, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23800",
            "ts": 0, "dur": 20,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::mul", "pid": 23803, "tid": "23800",
            "ts": 30, "dur": 101,
            "args": {"External id": 24505}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23805",
            "ts": 110, "dur": 18,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23803",
            "ts": 120, "dur": 36,
            "args": {"External id": 2516, "Input Dims": [[2049000]], "Input type": ["float"]}
            }
        ]
        """
        json_content1 = """[
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 20, "dur": 28,
            "args": {"External id": 256, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23805",
            "ts": 28, "dur": 30,
            "args": {"External id": 257, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:broadcast", "pid": 23803, "tid": "23803",
            "ts": 77, "dur": 6,
            "args": {"External id": 258, "Input Dims": [[53120]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::add_", "pid": 23803, "tid": "23800",
            "ts": 0, "dur": 30,
            "args": {"External id": 24504, "Input Dims": [[1000], [1000], []], "Input type": ["float", "float", "Int"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::mul", "pid": 23803, "tid": "23800",
            "ts": 70, "dur": 70,
            "args": {"External id": 24505}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23805",
            "ts": 88, "dur": 38,
            "args": {"External id": 2513, "Input Dims": [[2049000]], "Input type": ["float"]}
            },
            {
            "ph": "X", "cat": "cpu_op",
            "name": "gloo:all_reduce", "pid": 23803, "tid": "23803",
            "ts": 130, "dur": 16,
            "args": {"External id": 2516, "Input Dims": [[2049000]], "Input type": ["float"]}
            }
        ]
        """

        profile0 = parse_json_trace(json_content0, 'worker0')
        dist_data0 = DistributedRunProfileData(profile0)
        self.assertTrue(profile0.has_communication)
        self.assertEqual(len(profile0.comm_node_list), 5)
        self.assertEqual(profile0.steps_costs[0].costs, [0, 0, 0, 109, 0, 0, 47, 0, 156])

        profile1 = parse_json_trace(json_content1, 'worker1')
        dist_data1 = DistributedRunProfileData(profile1)
        self.assertTrue(profile1.has_communication)
        self.assertEqual(len(profile1.comm_node_list), 5)
        self.assertEqual(profile1.steps_costs[0].costs, [0, 0, 0, 98, 0, 0, 36, 12, 146])

        loader = RunLoader('test_gloo_cpu', '', None)
        dist_profile = loader._process_distributed_profiles([dist_data0, dist_data1], 0)
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker0'], [47, 74, 35, 0])
        self.assertEqual(dist_profile.steps_to_overlap['data']['0']['worker1'], [36, 64, 34, 12])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker0'], [75, 34])
        self.assertEqual(dist_profile.steps_to_wait['data']['0']['worker1'], [78, 20])
        self.assertEqual(dist_profile.comm_ops['data']['worker0']['rows'],
                         [['gloo:broadcast', 3, 637440, 212480, 63, 21, 41, 14],
                          ['gloo:all_reduce', 2, 16392000, 8196000, 46, 23, 34, 17]])
        self.assertEqual(dist_profile.comm_ops['data']['worker1']['rows'],
                         [['gloo:broadcast', 3, 637440, 212480, 44, 15, 44, 15],
                          ['gloo:all_reduce', 2, 16392000, 8196000, 54, 27, 34, 17]])

    def test_distributed_nccl_user_annotation_has_communication(self):
        # tests https://github.com/pytorch/kineto/issues/640
        json_content0 = """[
          {
            "ph": "X", "cat": "user_annotation", "name": "nccl:all_reduce", "pid": 128, "tid": 999,
            "ts": 1686070155939447, "dur": 71,
            "args": {
              "Trace name": "PyTorch Profiler", "Trace iteration": 0,
              "External id": 647,
              "Profiler Event Index": 134
            }
          }
        ]"""

        profile0 = parse_json_trace(json_content0, 'worker0')
        self.assertTrue(profile0.has_communication)

class TestMemoryCurve(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.event_data_cpu = [
            [1, 0, 0, 1, 4, 4, 0],          # alloc 1
            [20, 0, 0, 1, -4, 0, 0],        # free  1
            [100, 0, 0, 2, 8000, 8000, 0],  # alloc 2
            [200, 0, 0, 2, -8000, 0, 0],    # free  2
            [300, 0, 0, 3, 4, 4, 0],        # alloc 3
            [400, 0, 0, 4, 16, 20, 0],      # alloc 4
            [500, 0, 0, 5, 4000, 4020, 0],  # alloc 5
            [600, 0, 0, 4, -16, 4004, 0],   # free  4
            [700, 0, 0, 7, 80, 4084, 0],    # alloc 7
            [800, 0, 0, 3, -4, 4080, 0],    # free  3
            [900, 0, 0, 7, -80, 4000, 0],   # free  7
            [905, 0, 0, 4, -4000, 0, 0],    # free  5
        ]

        self.event_data_gpu = [
            [2, 1, 0, 11, 400, 400, 512],         # alloc 11
            [22, 1, 0, 11, -400, 0, 512],         # free  11
            [105, 1, 0, 12, 5000, 5000, 10240],   # alloc 12
            [106, 1, 0, 13, 3000, 8000, 10240],   # alloc 13
            [205, 1, 0, 12, -5000, 3000, 10240],  # free  12
            [401, 1, 0, 14, 1024, 4024, 10240],   # alloc 14
            [499, 1, 0, 15, 4, 4028, 10240],      # alloc 15
            [501, 1, 0, 13, -3000, 1028, 10240],  # free  13
            [502, 1, 0, 15, -4, 1024, 10240],     # free  15
            [906, 1, 0, 14, -1024, 0, 10240],     # free  14
        ]

        self.all_events = sorted(self.event_data_cpu + self.event_data_gpu, key=lambda e: e[0])

    def entry(self, ts, dev, dev_id, addr, alloc_size, total_allocated, total_reserved):
        return {
            'ph': 'i', 's': 't', 'name': '[memory]', 'pid': 0, 'tid': 0, 'ts': ts,
            'args': {
                'Device Type': dev,
                'Device Id': dev_id,
                'Addr': addr,
                'Bytes': alloc_size,
                'Total Allocated': total_allocated,
                'Total Reserved': total_reserved,
            },
        }

    def test_memory_curve_no_step_plot(self):
        json_content = json.dumps([self.entry(*data) for data in self.all_events])

        profile = parse_json_trace(json_content)
        profile.process()
        result = RunProfile.get_memory_curve(profile, time_metric='us', memory_metric='B', patch_for_step_plot=False)

        start_ts = profile.profiler_start_ts
        self.assertEqual(1, start_ts)

        curves = result['rows']

        self.assertIn('CPU', curves)
        self.assertIn('GPU0', curves)

        self.assertEqual(len(self.event_data_cpu), len(curves['CPU']))
        for i in range(len(self.event_data_cpu)):
            # adjusted timestamp
            self.assertEqual(self.event_data_cpu[i][0] - start_ts,  curves['CPU'][i][0])
            # total allocated
            self.assertEqual(self.event_data_cpu[i][-2], curves['CPU'][i][1])
            # total reserved
            self.assertEqual(self.event_data_cpu[i][-1], curves['CPU'][i][2])

        self.assertEqual(len(self.event_data_gpu), len(curves['GPU0']))
        for i in range(len(self.event_data_gpu)):
            self.assertEqual(self.event_data_gpu[i][0] - start_ts,  curves['GPU0'][i][0])
            self.assertEqual(self.event_data_gpu[i][-2], curves['GPU0'][i][1])
            self.assertEqual(self.event_data_gpu[i][-1], curves['GPU0'][i][2])

    def test_memory_curve_step_plot(self):
        json_content = json.dumps([self.entry(*data) for data in self.all_events])

        profile = parse_json_trace(json_content)
        profile.process()
        result = RunProfile.get_memory_curve(profile, time_metric='us', memory_metric='B', patch_for_step_plot=True)

        start_ts = profile.profiler_start_ts
        self.assertEqual(1, start_ts)

        curves = result['rows']

        self.assertIn('CPU', curves)
        self.assertIn('GPU0', curves)

        self.assertEqual(2 * len(self.event_data_cpu) - 1, len(curves['CPU']))
        for i in range(len(curves['CPU'])):
            if i % 2 == 0:  # original values
                # adjusted timestamp
                self.assertEqual(self.event_data_cpu[i//2][0] - start_ts,  curves['CPU'][i][0])
                # total allocated
                self.assertEqual(self.event_data_cpu[i//2][-2], curves['CPU'][i][1])
                # total reserved
                self.assertEqual(self.event_data_cpu[i//2][-1], curves['CPU'][i][2])
            else:  # interpolated values
                self.assertEqual(self.event_data_cpu[i//2+1][0] - start_ts,  curves['CPU'][i][0])
                self.assertEqual(self.event_data_cpu[i//2][-2], curves['CPU'][i][1])
                self.assertEqual(self.event_data_cpu[i//2][-1], curves['CPU'][i][2])

        self.assertEqual(2 * len(self.event_data_gpu) - 1, len(curves['GPU0']))
        for i in range(len(self.event_data_gpu)):
            if i % 2 == 0:  # original values
                self.assertEqual(self.event_data_gpu[i//2][0] - start_ts,  curves['GPU0'][i][0])
                self.assertEqual(self.event_data_gpu[i//2][-2], curves['GPU0'][i][1])
                self.assertEqual(self.event_data_gpu[i//2][-1], curves['GPU0'][i][2])
            else:  # interpolated values
                self.assertEqual(self.event_data_gpu[i//2+1][0] - start_ts,  curves['GPU0'][i][0])
                self.assertEqual(self.event_data_gpu[i//2][-2], curves['GPU0'][i][1])
                self.assertEqual(self.event_data_gpu[i//2][-1], curves['GPU0'][i][2])


class TestModuleView(unittest.TestCase):

    def test_build_module_hierarchy(self):
        from torch_tb_profiler.profiler import trace
        from torch_tb_profiler.profiler.module_op import (
            _build_module_hierarchy, aggegate_module_view)

        json_content = """[
            {
                "ph": "X",
                "cat": "python_function",
                "name": "test_root",
                "pid": 1908,
                "tid": 1908,
                "ts": 1,
                "dur": 19367,
                "args": {
                    "Python id": 1,
                    "Python thread": 0
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: MyModule",
                "pid": 1908,
                "tid": 1908,
                "ts": 2,
                "dur": 211,
                "args": {
                    "Python id": 2,
                    "Python parent id": 1,
                    "Python module id": 0
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: Linear",
                "pid": 1908,
                "tid": 1908,
                "ts": 5,
                "dur": 62,
                "args": {
                    "Python id": 3,
                    "Python parent id": 2,
                    "Python thread": 0,
                    "Python module id": 1
                }
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::addmm",
                "pid": 1908,
                "tid": 1908,
                "ts": 10,
                "dur": 31,
                "args": {
                    "External id": 12182,
                    "Fwd thread id": 0,
                    "Sequence number": 4006,
                    "python_caller_id": 3
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: MyModule",
                "pid": 1908,
                "tid": 1908,
                "ts": 1000,
                "dur": 211,
                "args": {
                    "Python id": 4,
                    "Python parent id": 1,
                    "Python module id": 0
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: Linear",
                "pid": 1908,
                "tid": 1908,
                "ts": 1001,
                "dur": 62,
                "args": {
                    "Python id": 5,
                    "Python parent id": 4,
                    "Python thread": 0,
                    "Python module id": 1
                }
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::addmm",
                "pid": 1908,
                "tid": 1908,
                "ts": 1002,
                "dur": 32,
                "args": {
                    "External id": 12182,
                    "Fwd thread id": 0,
                    "Sequence number": 4006,
                    "python_caller_id": 5
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: MyModule",
                "pid": 1908,
                "tid": 1908,
                "ts": 2000,
                "dur": 211,
                "args": {
                    "Python id": 6,
                    "Python parent id": 1,
                    "Python module id": 0
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: Linear",
                "pid": 1908,
                "tid": 1908,
                "ts": 2001,
                "dur": 62,
                "args": {
                    "Python id": 7,
                    "Python parent id": 6,
                    "Python thread": 0,
                    "Python module id": 1
                }
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::addmm",
                "pid": 1908,
                "tid": 1908,
                "ts": 2002,
                "dur": 33,
                "args": {
                    "External id": 12182,
                    "Fwd thread id": 0,
                    "Sequence number": 4006,
                    "python_caller_id": 7
                }
            },
            {
                "ph": "X",
                "cat": "python_function",
                "name": "nn.Module: Conv2",
                "pid": 1908,
                "tid": 1908,
                "ts": 3000,
                "dur": 211,
                "args": {
                    "Python id": 8,
                    "Python parent id": 1,
                    "Python module id": 100
                }
            }
        ]
        """
        data = parse_json_trace(json_content)
        stats = aggegate_module_view(data.tid2tree, data.events)
        stats.sort(key=lambda x: x.name)
        self.assertEqual(2, len(stats))
        self.assertEqual('Conv2', stats[0].name)
        self.assertEqual('MyModule', stats[1].name)
        self.assertEqual(1, len(stats[1].children))
        self.assertEqual('Linear', stats[1].children[0].name)

        content = json.loads(json_content)

        events = []
        for data in content:
            event = trace.create_event(data, False)
            events.append(event)

        roots = _build_module_hierarchy(events)
        roots.sort(key=lambda x: x.name)
        self.assertEqual(2, len(roots))
        self.assertEqual('nn.Module: Conv2', roots[0].name)
        self.assertEqual('nn.Module: MyModule', roots[1].name)
        self.assertEqual(1, len(roots[1].children))
        self.assertEqual('nn.Module: Linear', roots[1].children[0].name)


class TestDataPipe(unittest.TestCase):

    def test_datapipe(self):
        json_content = """[
            {
                "ph": "X", "cat": "cpu_op",
                "name": "enumerate(DataPipe)#ShufflerIterDataPipe", "pid": 7557, "tid": 7557,
                "ts": 100, "dur": 23,
                "args": {
                    "External id": 34
                }
            }
        ]"""
        profile = parse_json_trace(json_content)
        profile.process()

        dataloader_ranges = profile.role_ranges[ProfileRole.DataLoader]
        datapipe_range = None
        for range in dataloader_ranges:
            if range[0] == 100 and range[1] == 123:
                datapipe_range = range
                break
        self.assertTrue(datapipe_range is not None)

        root = next(iter(profile.tid2tree.values()))
        ops, _ = root.get_operator_and_kernels()
        datapipe_op = None
        for op in ops:
            if op.name.startswith('enumerate(DataPipe)'):
                datapipe_op = op
                break

        self.assertTrue(datapipe_op is None)


if __name__ == '__main__':
    unittest.main()
