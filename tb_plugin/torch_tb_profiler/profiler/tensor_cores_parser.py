# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from .. import consts
from .trace import EventTypes
from .tensor_core import TC_Allowlist


class TensorCoresParser:
    def __init__(self):
        self.tc_eligible_ops_kernel_ratio = 0.0
        # For calculating Tensor Cores time ratio.
        self.kernel_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.tc_ratio = [None] * consts.MAX_GPU_PER_NODE

    def parse_events(self, events, gpu_ids, steps_start_time, steps_end_time, tid2tree, ops):
        for event in events:
            if event.type == EventTypes.KERNEL:
                self._parse_event(event)
        self.tc_ratio = self._calculate_tc_ratio(gpu_ids, steps_start_time, steps_end_time)
        self.tc_eligible_ops_kernel_ratio = self._get_tc_eligible_ops_kernel_ratio(tid2tree, ops)

    def _parse_event(self, event):
        ts = event.ts
        dur = event.duration
        gpu_id = event.args.get("device", None)
        self.kernel_per_device[gpu_id].append((ts, ts + dur, event.name in TC_Allowlist))

    def _calculate_tc_ratio(self, gpu_ids, steps_start_time, steps_end_time):
        tc_ratio = [None] * consts.MAX_GPU_PER_NODE
        kernels_num = sum(len(self.kernel_per_device[gpu_id]) for gpu_id in gpu_ids)
        if kernels_num > 0: # If no kernel, then keep all self.tc_ratio as None.
            for gpu_id in gpu_ids:
                tc_time = 0
                total_time = 0
                kernels = self.kernel_per_device[gpu_id]
                for r in kernels:
                    min_time = max(r[0], steps_start_time)
                    max_time = min(r[1], steps_end_time)
                    if min_time < max_time:
                        dur = max_time - min_time
                        is_tc_used = r[2]
                        if is_tc_used:
                            tc_time += dur
                        total_time += dur
                if total_time > 0:
                    tc_ratio[gpu_id] = tc_time / total_time
                else:
                    tc_ratio[gpu_id] = 0.0
        return tc_ratio

    def _get_bottom_tc_eligible_operators(self, op_tree_node):
        ops = []
        for child in op_tree_node.children:
            child_ops = self._get_bottom_tc_eligible_operators(child)
            ops.extend(child_ops)
        # TC-eligible ops which have children TC-eligible ops will not be regarded as "bottom".
        if op_tree_node.tc_eligible and len(ops) == 0:
            ops.append(op_tree_node)
        return ops

    def _get_tc_eligible_ops_kernel_ratio(self, tid2tree, ops):
        def sum_self_kernel_time(ops):
            sum_time = 0
            for op in ops:
                for rt in op.runtimes:
                    # "CallTreeRoot" & "dummy" kernels are launched out of profiler step, so don't count them.
                    if not (op.name == "CallTreeRoot" and rt.name == "dummy"):
                        for k in rt.get_kernels():
                            sum_time += k.end_time - k.start_time
            return sum_time

        ops_bottom_tc_eligible = []
        for root in tid2tree.values():
            ops_bottom_tc_eligible.extend(self._get_bottom_tc_eligible_operators(root))
        ops_bottom_tc_eligible_kernel_sum = sum_self_kernel_time(ops_bottom_tc_eligible)
        ops_kernel_sum = sum_self_kernel_time(ops)
        tc_eligible_ops_kernel_ratio = ops_bottom_tc_eligible_kernel_sum / ops_kernel_sum \
            if ops_kernel_sum > 0 else 0.0
        return tc_eligible_ops_kernel_ratio
