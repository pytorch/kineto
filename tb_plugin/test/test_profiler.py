import json
import gzip
import unittest

import torch_tb_profiler.profiler.trace as trace
from torch_tb_profiler.profiler.data import RunProfileData
from torch_tb_profiler.profiler.overall_parser import ProfileRole
from torch_tb_profiler.run import RunProfile

SCHEMA_VERSION = 1
WORKER_NAME = "worker0"


def parse_json_trace(json_content):
    trace_json = json.loads(json_content)
    trace_json = {"schemaVersion": 1, "traceEvents": trace_json}
    profile = RunProfileData(WORKER_NAME)
    profile.trace_json = trace_json
    profile.events = []
    for data in trace_json["traceEvents"]:
        event = trace.create_event(data)
        if event is not None:
            profile.events.append(event)
    return profile


'''
All the events in json string are only simulation, not actual generated events.
We removed the data fields that not used by current version of our profiler,
for easy to check correctness and shorter in length. 
We even renamed the data values such as kernel name or "ts", to simplify the string.
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
                if op_agg.name == "aten::to":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[[2, 8, 5], [], [], [], [], [], [], []]")
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 0)
                    self.assertEqual(op_agg.self_host_duration, 60)
                    self.assertEqual(op_agg.self_device_duration, 0)
                if op_agg.name == "aten::nll_loss_backward":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[[], [32, 1000], [32], [], [], [], []]")
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 70)
                    self.assertEqual(op_agg.device_duration, 30)
                    self.assertEqual(op_agg.self_host_duration, 70 - 20 - 10 - 5)
                    self.assertEqual(op_agg.self_device_duration, 30)
            self.assertEqual(op_count, 2)

        test_op_list(profile.op_list_groupby_name)
        test_op_list(profile.op_list_groupby_name_input)

        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)
        self.assertEqual(profile.kernel_stat.shape[0], 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].name,
                         "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].op_name, "aten::nll_loss_backward")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].total_duration, 15)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].min_duration, 15)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].max_duration, 15)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]["sum"], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]["mean"], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]["min"], 15)
        self.assertEqual(profile.kernel_stat.iloc[0]["max"], 15)

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
            if op_agg.name == "aten::mat_mul":
                op_count += 1
                self.assertEqual(op_agg.device_duration, 5 + 6 + 7 + 8)
                self.assertEqual(op_agg.self_device_duration, 6 + 8)
            if op_agg.name == "aten::mm":
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
            if op_agg.name == "aten::mat_mul":
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 100 - 70)
            if op_agg.name == "aten::mm":
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
            if op_agg.name == "aten::mat_mul":
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 100 - 70)
            if op_agg.name == "aten::mm":
                op_count += 1
                self.assertEqual(op_agg.self_host_duration, 70)
        self.assertEqual(op_count, 2)

    # Test multiple father-child operators with same name.
    # In this case, all the operators except the top operator should be removed,
    # and all runtime/kernels belong to the children operators should be attached to the only kept one.
    # This behavior is to keep consistent with _remove_dup_nodes in torch/autograd/profiler.py.
    def test_remove_dup_nodes(self):
        json_content = """
          [{
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
          }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(len(profile.op_list_groupby_name), 1)
        self.assertEqual(profile.op_list_groupby_name[0].self_device_duration, 8)

    # Test Runtime with "external id" 0.
    # This kind of Runtime should not be attached to any operator,
    # and should be included in accumulating device time.
    def test_top_level_runtime(self):
        # This operator is different thread with the runtime.
        json_content = """
          [{
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "123",
            "ts": 100, "dur": 100,
            "args": {"Input Dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Runtime", 
            "name": "cudaLaunchKernel", "pid": 13721, "tid": "456",
            "ts": 130, "dur": 20,
            "args": {"correlation": 335, "external id": 0}
          },
          {
            "ph": "X", "cat": "Kernel", 
            "name": "void gemmSN_TN_kernel_64addr", "pid": 0, "tid": "stream 7",
            "ts": 220, "dur": 8,
            "args": {"correlation": 335, "external id": 0, "device": 0}
          }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(profile.op_list_groupby_name[0].device_duration, 0)
        self.assertEqual(profile.op_list_groupby_name[0].self_device_duration, 0)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)

    # Test Runtime directly called in ProfilerStep, not inside any operator.
    def test_runtime_called_by_profilerstep(self):
        json_content = """
          [{
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
          }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Kernel], 8)
        self.assertEqual(step.costs[ProfileRole.Runtime], 20)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 0)
        self.assertEqual(step.costs[ProfileRole.Other], 300 - 8 - 20)
        self.assertEqual(len(profile.op_list_groupby_name), 0)  # ProfilerStep is not regarded as an operator.
        self.assertEqual(len(profile.op_list_groupby_name_input), 0)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

    # Test one Runtime lauch more than one Kernels.
    # Sometimes such as running Bert using DataParallel mode(1 process, 2GPUs),
    # one runtime such as cudaLaunchCooperativeKernelMultiDevice could trigger more than one kernel,
    # each Kernel runs at a seperate GPU card.
    def test_runtime_launch_multipe_kernels(self):
        json_content = """
          [{
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
          }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        self.assertEqual(profile.op_list_groupby_name[0].device_duration, 120318 + 132800)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 2)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

    # Test when there is no ProfilerStep#.
    def test_no_profilerstep(self):
        json_content = """
          [{
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
          }]
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
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)
        self.assertEqual(len(profile.kernel_list_groupby_name_op), 1)

        def test_op_list(op_list):
            op_count = 0
            for op_agg in op_list:
                if op_agg.name == "aten::to":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[[2, 8, 5], [], [], [], [], [], [], []]")
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 0)
                    self.assertEqual(op_agg.self_host_duration, 60)
                    self.assertEqual(op_agg.self_device_duration, 0)
                if op_agg.name == "aten::nll_loss_backward":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[[], [32, 1000], [32], [], [], [], []]")
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 70)
                    self.assertEqual(op_agg.device_duration, 100)
                    self.assertEqual(op_agg.self_host_duration, 70 - 20)
                    self.assertEqual(op_agg.self_device_duration, 100)
            self.assertEqual(op_count, 2)

        test_op_list(profile.op_list_groupby_name)
        test_op_list(profile.op_list_groupby_name_input)

        self.assertEqual(profile.kernel_list_groupby_name_op[0].name,
                         "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].op_name, "aten::nll_loss_backward")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].total_duration, 100)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].min_duration, 100)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].max_duration, 100)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]["sum"], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]["mean"], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]["min"], 100)
        self.assertEqual(profile.kernel_stat.iloc[0]["max"], 100)

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
        self.assertEqual(step.costs[ProfileRole.Total], 320 - 100)  # Device side takes effect.
        step = profile.steps_costs[1]
        self.assertEqual(step.costs[ProfileRole.Kernel], 200)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], 50 - 5)
        self.assertEqual(step.costs[ProfileRole.Other], 360 - 350)
        self.assertEqual(step.costs[ProfileRole.Total], 610 - 350)  # Device side takes effect.
        self.assertEqual(profile.avg_costs.costs[ProfileRole.Total], ((320 - 100) + (610 - 350)) / 2)

        self.assertEqual(len(profile.op_list_groupby_name), 2)
        self.assertEqual(len(profile.op_list_groupby_name_input), 2)

        def test_op_list(op_list):
            op_count = 0
            for op_agg in op_list:
                if op_agg.name == "aten::to":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[[2, 8, 5], [], [], [], [], [], [], []]")
                    self.assertEqual(op_agg.calls, 1)
                    self.assertEqual(op_agg.host_duration, 60)
                    self.assertEqual(op_agg.device_duration, 40)
                    self.assertEqual(op_agg.self_host_duration, 60 - 5)
                    self.assertEqual(op_agg.self_device_duration, 40)
                if op_agg.name == "aten::mm":
                    op_count += 1
                    self.assertEqual(op_agg.input_shape, "[]")
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
                         "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].op_name, "aten::mm")
        self.assertEqual(profile.kernel_list_groupby_name_op[0].calls, 1)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].total_duration, 200)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].min_duration, 200)
        self.assertEqual(profile.kernel_list_groupby_name_op[0].max_duration, 200)
        self.assertEqual(profile.kernel_stat.iloc[0]["count"], 1)
        self.assertEqual(profile.kernel_stat.iloc[0]["sum"], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]["mean"], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]["min"], 200)
        self.assertEqual(profile.kernel_stat.iloc[0]["max"], 200)

    # Test self time and total time on operator with nested operator.
    def test_self_time(self):
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
            profile.process()

            op_count = 0
            for op_agg in profile.op_list_groupby_name:
                if op_agg.name == "aten::mat_mul":
                    op_count += 1
                    self.assertEqual(op_agg.host_duration, 100)
                    self.assertEqual(op_agg.device_duration, 20 + 16)
                    self.assertEqual(op_agg.self_host_duration, 100 - 40)
                    self.assertEqual(op_agg.self_device_duration, 16)
                if op_agg.name == "aten::mm":
                    op_count += 1
                    self.assertEqual(op_agg.host_duration, 40)
                    self.assertEqual(op_agg.device_duration, 20)
                    self.assertEqual(op_agg.self_host_duration, 40)
                    self.assertEqual(op_agg.self_device_duration, 20)
            self.assertEqual(op_count, 2)

    # 2 steps with overlap with each other.
    def test_multiple_profilersteps_with_overlap(self):
        # The kernel with "correlation" as 123 is launched by previous step,
        # its end time is bigger than "ProfilerStep#1"'s start time,
        # so it is regarded as beginning of "ProfilerStep#1".
        # The memcpy with "correlation" as 334 is launched by "ProfilerStep#1",
        # its end time is bigger than "ProfilerStep#2"'s start time,
        # so it is regarded as beginning of "ProfilerStep#2".
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
            "args": {"correlation": 123, "external id": 0, "device": 0}
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
        self.assertEqual(step.costs[ProfileRole.CpuOp], (200 + 60) - (150 + 90) - 5)
        self.assertEqual(step.costs[ProfileRole.Other], 280 - (200 + 60))
        self.assertEqual(step.costs[ProfileRole.Total], (280 + 100) - (150 + 90))  # Device side takes effect.
        step = profile.steps_costs[1]
        self.assertEqual(step.costs[ProfileRole.Kernel], 200)
        self.assertEqual(step.costs[ProfileRole.Memcpy], 0)
        self.assertEqual(step.costs[ProfileRole.Memset], 0)
        self.assertEqual(step.costs[ProfileRole.Runtime], 5)
        self.assertEqual(step.costs[ProfileRole.DataLoader], 0)
        self.assertEqual(step.costs[ProfileRole.CpuOp], (280 + 100) - 360 + (410 - 405))
        self.assertEqual(step.costs[ProfileRole.Other], 0)
        self.assertEqual(step.costs[ProfileRole.Total], 610 - (280 + 100))  # Device side takes effect.

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
            "args": {"correlation": 123, "external id": 0, "device": 0}
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

        self.assertEqual(len(profile.steps_costs), 1)  # The last 2 steps without kernels are removed from overall view.
        step = profile.steps_costs[0]
        self.assertEqual(step.costs[ProfileRole.Total], (150 + 180) - (90 + 20))

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

        self.assertEqual(len(profile.gpu_ids), 1)
        self.assertAlmostEqual(profile.gpu_utilization[1], (40 + 20) / 120)
        self.assertAlmostEqual(profile.sm_efficency[1],
                               (0.5 * (135 - 130)
                                + 1.0 * (140 - 135)
                                + 0.6 * (145 - 140)
                                + 0.9 * (150 - 145)
                                + 0.3 * (170 - 150)
                                + 1.0 * (220 - 200)) / (220 - 100))
        self.assertAlmostEqual(profile.occupancy[1],
                               (0.6 * 10 + 0.1 * 15 + 1.0 * 25 + 0.3 * 20) / (10 + 15 + 25 + 20))

        gpu_util_expected = [(100, 0), (110, 0), (120, 0), (130, 1.0), (140, 1.0), (150, 1.0), (160, 1.0),
                             (170, 0), (180, 0), (190, 0), (200, 1.0), (210, 1.0), (220, 0)]
        for gpu_id in profile.gpu_ids:
            buckets = profile.gpu_util_buckets[gpu_id]
            gpu_util_id = 0
            for b in buckets:
                self.assertEqual(b[0], gpu_util_expected[gpu_util_id][0])
                self.assertAlmostEqual(b[1], gpu_util_expected[gpu_util_id][1])
                gpu_util_id += 1
            self.assertEqual(gpu_util_id, len(gpu_util_expected))

        sm_efficiency_expected = [(130, 0.5), (135, 0), (135, 1.0), (140, 0), (140, 0.6), (145, 0), (145, 0.9),
                                  (150, 0), (150, 0.3), (170, 0), (170, 0), (200, 0), (200, 1.0), (220, 0)]
        for gpu_id in profile.gpu_ids:
            ranges = profile.approximated_sm_efficency_ranges[gpu_id]
            sm_efficiency_id = 0
            for r in ranges:
                self.assertEqual(r[0][0], sm_efficiency_expected[sm_efficiency_id][0])
                self.assertAlmostEqual(r[1], sm_efficiency_expected[sm_efficiency_id][1])
                sm_efficiency_id += 1
                self.assertEqual(r[0][1], sm_efficiency_expected[sm_efficiency_id][0])
                self.assertAlmostEqual(0, sm_efficiency_expected[sm_efficiency_id][1])
                sm_efficiency_id += 1
            self.assertEqual(sm_efficiency_id, len(sm_efficiency_expected))

        count = 0
        for agg_by_op in profile.kernel_list_groupby_name_op:
            if agg_by_op.name == "void gemmSN_TN_kernel_64addr" and agg_by_op.op_name == "aten::mat_mul":
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, 0.6)
                self.assertAlmostEqual(agg_by_op.avg_occupancy, 0.1)
                count += 1
            if agg_by_op.name == "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>" and \
                    agg_by_op.op_name == "aten::mm":
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, (0.5 * 10 + 0.3 * 25) / (10 + 25))
                self.assertAlmostEqual(agg_by_op.avg_occupancy, (0.6 * 10 + 1.0 * 25) / (10 + 25))
                count += 1
            if agg_by_op.name == "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>" and \
                    agg_by_op.op_name == "aten::mat_mul":
                self.assertAlmostEqual(agg_by_op.avg_blocks_per_sm, 10.5)
                self.assertAlmostEqual(agg_by_op.avg_occupancy, 0.3)
                count += 1
        self.assertEqual(count, 3)

        count = 0
        for _id, (name, row) in enumerate(profile.kernel_stat.iterrows()):
            # The kernel with zero "dur" should be ignored.
            if name == "void gemmSN_TN_kernel_64addr":
                self.assertAlmostEqual(row["blocks_per_sm"], 0.6)
                self.assertAlmostEqual(row["occupancy"], 0.1)
                count += 1
            if name == "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>":
                self.assertAlmostEqual(row["blocks_per_sm"], (0.5 * 10 + 0.3 * 25 + 10.5 * 20) / (10 + 25 + 20))
                self.assertAlmostEqual(row["occupancy"], (0.6 * 10 + 1.0 * 25 + 0.3 * 20) / (10 + 25 + 20))
                count += 1
        self.assertEqual(count, 2)

    def test_dump_gpu_metrics(self):
        profile = RunProfile("test_dump_gpu_metrics")
        # Faked data for easy to see in UI. Real data values are 1/100 of these.
        profile.gpu_util_buckets = [[(1621401187223005, 0.0), (1621401187224005, 0.0),
                                     (1621401187225005, 0.6), (1621401187226005, 0.5),
                                     (1621401187227005, 0.6), (1621401187228005, 0.2),
                                     (1621401187229005, 0.6), (1621401187230005, 0.1),
                                     (1621401187231005, 0.5), (1621401187232005, 0.2),
                                     (1621401187233005, 0.3), (1621401187234005, 0.4),
                                     (1621401187235005, 0.4219409282700422),
                                     (1621401187236901, 0)]]
        # Faked data for easy to see in UI. Real data values are 1/10 of these.
        profile.approximated_sm_efficency_ranges = \
            [[((1621401187225275, 1621401187225278), 0.25), ((1621401187225530, 1621401187225532), 0.125),
              ((1621401187225820, 1621401187225821), 0.125), ((1621401187226325, 1621401187226327), 0.25),
              ((1621401187226575, 1621401187226577), 0.125), ((1621401187226912, 1621401187226913), 0.125),
              ((1621401187227092, 1621401187227094), 0.125), ((1621401187227619, 1621401187227620), 0.125),
              ((1621401187227745, 1621401187227746), 0.125), ((1621401187227859, 1621401187227860), 0.125),
              ((1621401187227973, 1621401187227974), 0.125), ((1621401187228279, 1621401187228280), 0.125),
              ((1621401187228962, 1621401187228963), 0.125), ((1621401187229153, 1621401187229155), 0.125),
              ((1621401187229711, 1621401187229715), 0.125), ((1621401187230162, 1621401187230163), 0.125),
              ((1621401187231100, 1621401187231103), 0.125), ((1621401187231692, 1621401187231694), 0.5),
              ((1621401187232603, 1621401187232604), 0.125), ((1621401187232921, 1621401187232922), 0.125),
              ((1621401187233342, 1621401187233343), 0.125), ((1621401187233770, 1621401187233772), 0.125),
              ((1621401187234156, 1621401187234159), 0.125), ((1621401187234445, 1621401187234446), 0.125),
              ((1621401187235025, 1621401187235028), 0.125), ((1621401187235555, 1621401187235556), 0.125),
              ((1621401187236158, 1621401187236159), 0.125), ((1621401187236278, 1621401187236279), 0.125),
              ((1621401187236390, 1621401187236391), 0.125), ((1621401187236501, 1621401187236502), 0.125)]]

        trace_json_flat_path = "gpu_metrics_input.json"
        with open(trace_json_flat_path, "rb") as file:
            raw_data = file.read()
        data_with_gpu_metrics_compressed = profile.append_gpu_metrics(raw_data)
        data_with_gpu_metrics_flat = gzip.decompress(data_with_gpu_metrics_compressed)

        trace_json_expected_path = "gpu_metrics_expected.json"
        with open(trace_json_expected_path, "rb") as file:
            data_expected = file.read()

        # Parse to json in order to ignore text format difference.
        data_with_gpu_metrics_json = json.loads(data_with_gpu_metrics_flat.decode("utf8"))
        data_expected_json = json.loads(data_expected.decode("utf8"))
        data_with_gpu_metrics_str = json.dumps(data_with_gpu_metrics_json, sort_keys=True)
        data_expected_str = json.dumps(data_expected_json, sort_keys=True)

        self.assertEqual(data_with_gpu_metrics_str, data_expected_str)

        try:
            data = json.loads(data_with_gpu_metrics_flat.decode("utf8"))
        except:
            self.assertTrue(False, "The string fails to be parsed by json after appending gpu metrics.")


if __name__ == '__main__':
    unittest.main()
