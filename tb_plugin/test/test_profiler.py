import json
import unittest

import torch_tb_profiler.profiler.trace as trace
from torch_tb_profiler.profiler.data import RunProfileData

SCHEMA_VERSION = 1
WORKER_NAME = "worker0"


def parse_json_trace(json_content):
    trace_json = json.loads(json_content)
    profile = RunProfileData(WORKER_NAME)
    parser = trace.get_event_parser(SCHEMA_VERSION)
    profile.events = []
    for data in trace_json:
        event = parser.parse(data)
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
            "args": {"Input dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::to", "pid": 13721, "tid": "123",
            "ts": 200, "dur": 60,
            "args": {"Input dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::nll_loss_backward", "pid": 13721, "tid": "456",
            "ts": 340, "dur": 70,
            "args": {"Input dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
            "ts": 50, "dur": 400,
            "args": {"Input dims": [], "External id": 1}
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
            "args": {"correlation": 40348, "external id": 4}
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
        self.assertEqual(step.kernel_cost, 15)
        self.assertEqual(step.memcpy_cost, 10)
        self.assertEqual(step.memset_cost, 5)
        self.assertEqual(step.runtime_cost, 30)
        self.assertEqual(step.dataloader_cost, 180)
        self.assertEqual(step.cpuop_cost, 35)
        self.assertEqual(step.other_cost, 125)

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
            "args": {"Input dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "456",
            "ts": 120, "dur": 70,
            "args": {"Input dims": [], "External id": 4}
          },
          {
            "ph": "X", "cat": "Kernel", 
            "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
            "ts": 130, "dur": 5,
            "args": {"correlation": 334, "external id": 4}
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
            "args": {"correlation": 335, "external id": 2}
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
            "args": {"correlation": 336, "external id": 4}
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
            "args": {"correlation": 337, "external id": 2}
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
            "args": {"Input dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "456",
            "ts": 100, "dur": 70,
            "args": {"Input dims": [], "External id": 4}
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
            "args": {"Input dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "456",
            "ts": 130, "dur": 70,
            "args": {"Input dims": [], "External id": 4}
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
            "args": {"Input dims": [], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "456",
            "ts": 110, "dur": 80,
            "args": {"Input dims": [], "External id": 3}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "456",
            "ts": 120, "dur": 60,
            "args": {"Input dims": [], "External id": 4}
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
            "args": {"correlation": 335, "external id": 4}
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
            "args": {"Input dims": [], "External id": 2}
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
            "args": {"correlation": 335, "external id": 0}
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
            "args": {"Input dims": [], "External id": 2}
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
            "args": {"correlation": 335, "external id": 2}
          }]
        """
        profile = parse_json_trace(json_content)
        profile.process()
        step = profile.steps_costs[0]
        self.assertEqual(step.kernel_cost, 8)
        self.assertEqual(step.runtime_cost, 20)
        self.assertEqual(step.cpuop_cost, 0)
        self.assertEqual(step.other_cost, 300 - 8 - 20)
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
            "args": {"Input dims": [], "External id": 2}
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
                     "correlation": 335, "external id": 2}
          },
          {
            "ph": "X", "cat": "Kernel", 
            "name": "ncclBroadcastRingLLKernel_copy_i8(ncclColl)", "pid": 0, "tid": "stream 22",
            "ts": 170, "dur": 132800,
            "args": {"device": 1, "context": 2, "stream": 22,
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
            "args": {"Input dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 3}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::nll_loss_backward", "pid": 13721, "tid": "456",
            "ts": 300, "dur": 70,
            "args": {"Input dims": [[], [32, 1000], [32], [], [], [], []], "External id": 4}
          },    
          {
            "ph": "X", "cat": "Kernel", 
            "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
            "ts": 320, "dur": 100,
            "args": {"correlation": 40348, "external id": 4}
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
        self.assertEqual(step.kernel_cost, 100)
        self.assertEqual(step.memcpy_cost, 0)
        self.assertEqual(step.memset_cost, 0)
        self.assertEqual(step.runtime_cost, 320 - 310)
        self.assertEqual(step.dataloader_cost, 0)
        self.assertEqual(step.cpuop_cost, 60 + (310 - 300))
        # If no ProfilerStep, all events will be regarded as a step.
        self.assertEqual(step.other_cost, 300 - (100 + 60))
        self.assertEqual(step.step_total_cost, (320 + 100) - 100)
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

    def test_multiple_profilersteps(self):
        json_content = """
          [{
            "ph": "X", "cat": "Operator", 
            "name": "ProfilerStep#1", "pid": 13721, "tid": "123",
            "ts": 100, "dur": 200,
            "args": {"Input dims": [], "External id": 1}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::to", "pid": 13721, "tid": "123",
            "ts": 200, "dur": 60,
            "args": {"Input dims": [[2, 8, 5], [], [], [], [], [], [], []], "External id": 2}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "ProfilerStep#2", "pid": 13721, "tid": "123",
            "ts": 350, "dur": 150,
            "args": {"Input dims": [], "External id": 3}
          },
          {
            "ph": "X", "cat": "Operator", 
            "name": "aten::mm", "pid": 13721, "tid": "123",
            "ts": 360, "dur": 50,
            "args": {"Input dims": [], "External id": 4}
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
            "args": {"correlation": 40348, "external id": 4}
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
        self.assertEqual(step.kernel_cost, 0)
        self.assertEqual(step.memcpy_cost, (100 + 200) - 280)
        self.assertEqual(step.memset_cost, 0)
        self.assertEqual(step.runtime_cost, 5)
        self.assertEqual(step.dataloader_cost, 0)
        self.assertEqual(step.cpuop_cost, 60 - 5)
        self.assertEqual(step.other_cost, 200 - 60 - 20)
        self.assertEqual(step.step_total_cost, 200)  # Only the time inside ProfilerStep will count.
        step = profile.steps_costs[1]
        self.assertEqual(step.kernel_cost, (350 + 150) - 410)
        self.assertEqual(step.memcpy_cost, 0)
        self.assertEqual(step.memset_cost, 0)
        self.assertEqual(step.runtime_cost, 5)
        self.assertEqual(step.dataloader_cost, 0)
        self.assertEqual(step.cpuop_cost, 50 - 5)
        self.assertEqual(step.other_cost, 360 - 350)
        self.assertEqual(step.step_total_cost, 150)  # Only the time inside ProfilerStep will count.
        self.assertEqual(profile.avg_costs.step_total_cost, (200 + 150) / 2)

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
                "args": {"Input dims": [], "External id": 2}
              },
              {
                "ph": "X", "cat": "Operator", 
                "name": "aten::mm", "pid": 13721, "tid": "456",
                "ts": 120, "dur": 40,
                "args": {"Input dims": [], "External id": 4}
              },
              {
                "ph": "X", "cat": "Kernel", 
                "name": "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>", "pid": 0, "tid": "stream 7",
                "ts": 155, "dur": 20,
                "args": {"correlation": 334, "external id": 4}
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
                "args": {"correlation": 335, "external id": 2}
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


if __name__ == '__main__':
    unittest.main()
