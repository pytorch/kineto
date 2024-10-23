# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from typing import Dict, Iterable, List

from .. import consts
from .node import OperatorNode


class TensorCoresParser:
    def __init__(self, tc_ratio: List[float], tc_eligible_ops_kernel_ratio: float):
        # For calculating Tensor Cores time ratio per GPU.
        self.tc_ratio = tc_ratio
        self.tc_eligible_ops_kernel_ratio = tc_eligible_ops_kernel_ratio

    @classmethod
    def parse_events(cls, tid2tree: Dict[str, OperatorNode], ops: Iterable[OperatorNode], gpu_ids: Iterable[int]):
        tc_ratio = cls._calculate_tc_ratio(ops, gpu_ids)
        tc_eligible_ops_kernel_ratio = cls._get_tc_eligible_ops_kernel_ratio(tid2tree, ops)
        return cls(tc_ratio, tc_eligible_ops_kernel_ratio)

    @staticmethod
    def _calculate_tc_ratio(ops: Iterable[OperatorNode], gpu_ids: Iterable[int]):
        tc_ratio: List[float] = [None] * consts.MAX_GPU_PER_NODE
        tc_time = [0] * consts.MAX_GPU_PER_NODE
        total_time = [0] * consts.MAX_GPU_PER_NODE
        has_kernel = False
        for op in ops:
            for rt in op.runtimes:
                # 'CallTreeRoot' & 'dummy' kernels are launched out of profiler step, so don't count them.
                if not (op.name == 'CallTreeRoot' and rt.name == 'dummy'):
                    for k in rt.get_kernels():
                        has_kernel = True
                        dur = k.end_time - k.start_time
                        is_tc_used = k.tc_used
                        if is_tc_used:
                            tc_time[k.device_id] += dur
                        total_time[k.device_id] += dur
        if has_kernel:  # If no kernel, then keep all self.tc_ratio as None.
            for gpu_id in gpu_ids:
                if total_time[gpu_id] > 0:
                    tc_ratio[gpu_id] = tc_time[gpu_id] / total_time[gpu_id]
                else:
                    tc_ratio[gpu_id] = 0.0
        return tc_ratio

    @staticmethod
    def _get_bottom_tc_eligible_operators(op_tree_node: OperatorNode):
        ops: List[OperatorNode] = []
        for child in op_tree_node.children:
            child_ops = TensorCoresParser._get_bottom_tc_eligible_operators(child)
            ops.extend(child_ops)
        # TC-eligible ops which have children TC-eligible ops will not be regarded as 'bottom'.
        if op_tree_node.tc_eligible and len(ops) == 0:
            ops.append(op_tree_node)
        return ops

    @staticmethod
    def _get_tc_eligible_ops_kernel_ratio(tid2tree: Dict[int, OperatorNode], ops: Iterable[OperatorNode]):
        def sum_self_kernel_time(ops: Iterable[OperatorNode]):
            sum_time = 0
            for op in ops:
                for rt in op.runtimes:
                    # 'CallTreeRoot' & 'dummy' kernels are launched out of profiler step, so don't count them.
                    if not (op.name == 'CallTreeRoot' and rt.name == 'dummy'):
                        for k in rt.get_kernels():
                            sum_time += k.end_time - k.start_time
            return sum_time

        ops_bottom_tc_eligible = []
        for root in tid2tree.values():
            ops_bottom_tc_eligible.extend(TensorCoresParser._get_bottom_tc_eligible_operators(root))
        ops_bottom_tc_eligible_kernel_sum = sum_self_kernel_time(ops_bottom_tc_eligible)
        ops_kernel_sum = sum_self_kernel_time(ops)
        tc_eligible_ops_kernel_ratio = ops_bottom_tc_eligible_kernel_sum / ops_kernel_sum \
            if ops_kernel_sum > 0 else 0.0
        return tc_eligible_ops_kernel_ratio
