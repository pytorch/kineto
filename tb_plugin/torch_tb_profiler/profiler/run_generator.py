# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from collections import OrderedDict
from typing import Dict, Iterable, List

from .. import consts, utils
from ..run import DistributedRunProfile, RunProfile
from .data import DistributedRunProfileData, RunProfileData
from .module_op import aggegate_module_view, aggegate_pl_module_view
from .op_agg import KernelAggByNameOp, OperatorAgg
from .overall_parser import ProfileRole

logger = utils.get_logger()


class RunGenerator:
    def __init__(self, worker, span, profile_data: RunProfileData):
        self.worker = worker
        self.span = span
        self.profile_data = profile_data

    def generate_run_profile(self):
        profile_run = RunProfile(self.worker, self.span)
        profile_run.is_pytorch_lightning = self.profile_data.is_pytorch_lightning
        profile_run.has_runtime = self.profile_data.has_runtime
        profile_run.has_kernel = self.profile_data.has_kernel
        profile_run.has_communication = self.profile_data.has_communication
        profile_run.has_memcpy_or_memset = self.profile_data.has_memcpy_or_memset
        profile_run.profiler_start_ts = self.profile_data.profiler_start_ts
        profile_run.views.append(consts.OVERALL_VIEW)
        profile_run.overview = self._generate_overview()

        profile_run.views.append(consts.OP_VIEW)
        profile_run.operation_pie_by_name = self._generate_op_pie()
        profile_run.operation_table_by_name = self._generate_op_table(self.profile_data.op_list_groupby_name)
        profile_run.operation_stack_by_name = self._generate_op_table_for_stack(False)
        profile_run.operation_pie_by_name_input = self._generate_op_pie(True)
        profile_run.operation_table_by_name_input = self._generate_op_table(
            self.profile_data.op_list_groupby_name_input, True)
        profile_run.operation_stack_by_name_input = self._generate_op_table_for_stack(True)

        if self.profile_data.has_kernel:
            profile_run.views.append(consts.KERNEL_VIEW)
            profile_run.kernel_op_table = self._generate_kernel_op_table()
            profile_run.kernel_pie = self._generate_kernel_pie()
            profile_run.kernel_table = self._generate_kernel_table()
            profile_run.tc_pie = self._generate_tc_pie()

        profile_run.views.append(consts.TRACE_VIEW)
        profile_run.trace_file_path = self.profile_data.trace_file_path

        profile_run.gpu_metrics = self.profile_data.gpu_metrics_parser.get_gpu_metrics()

        gpu_infos = {gpu_id: RunGenerator._get_gpu_info(self.profile_data.device_props, gpu_id)
                     for gpu_id in self.profile_data.gpu_metrics_parser.gpu_ids}
        gpu_infos = {gpu_id: gpu_info for gpu_id, gpu_info in gpu_infos.items() if gpu_info is not None}

        profile_run.gpu_summary, profile_run.gpu_tooltip = \
            self.profile_data.gpu_metrics_parser.get_gpu_metrics_data_tooltip(
                gpu_infos, self.profile_data.tc_ratio)

        profile_run.tid2tree = self.profile_data.tid2tree
        profile_run.pl_tid2tree = self.profile_data.pl_tid2tree

        if self.profile_data.memory_snapshot:
            profile_run.views.append(consts.MEMORY_VIEW)
            profile_run.memory_snapshot = self.profile_data.memory_snapshot

        profile_run.module_stats = aggegate_module_view(self.profile_data.tid2tree, self.profile_data.events)
        profile_run.pl_module_stats = aggegate_pl_module_view(self.profile_data.tid2tree, self.profile_data.events)
        if profile_run.is_pytorch_lightning and profile_run.pl_module_stats:
            profile_run.views.append(consts.LIGHTNING_VIEW)
        elif profile_run.module_stats:
            profile_run.views.append(consts.MODULE_VIEW)

        return profile_run

    def _generate_overview(self):
        def build_part_time_str(part_cost: float, part_name: str):
            format_str = ('<div class="visualization-tooltip" style="white-space: nowrap;">'
                          'Step {}<br>'
                          'Total: {}us<br>'
                          '<b>{}: {}us</b><br>'
                          'Percentage: {}%'
                          '</div>')
            percentage = round(100 * part_cost / costs.costs[ProfileRole.Total], 2)
            return format_str.format(step_name, costs.costs[ProfileRole.Total], part_name, part_cost, percentage)

        def build_avg_cost_dict(part_name: str, part_cost: float):
            cost_dict = {'name': part_name,
                         'description': '',
                         'value': round(part_cost),
                         'extra': round(100 * part_cost / self.profile_data.avg_costs.costs[ProfileRole.Total], 2)}
            return cost_dict

        show_gpu = (self.profile_data.has_runtime
                    or self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset)

        column_tootip = {'type': 'string', 'role': 'tooltip', 'p': {'html': 'true'}}
        data = {}
        data['steps'] = {}
        data['steps']['columns'] = [{'type': 'string', 'name': 'Step'}]
        if show_gpu:
            data['steps']['columns'].extend([{'type': 'number', 'name': 'Kernel'},
                                             column_tootip,
                                             {'type': 'number', 'name': 'Memcpy'},
                                             column_tootip,
                                             {'type': 'number', 'name': 'Memset'},
                                             column_tootip])
        if self.profile_data.has_communication:
            data['steps']['columns'].extend([{'type': 'number', 'name': 'Communication'},
                                             column_tootip])
        if show_gpu:
            data['steps']['columns'].extend([{'type': 'number', 'name': 'Runtime'},
                                             column_tootip])
        data['steps']['columns'].extend([{'type': 'number', 'name': 'DataLoader'},
                                         column_tootip,
                                         {'type': 'number', 'name': 'CPU Exec'},
                                         column_tootip,
                                         {'type': 'number', 'name': 'Other'},
                                         column_tootip])

        data['steps']['rows'] = []
        for i in range(len(self.profile_data.steps_costs)):
            costs = self.profile_data.steps_costs[i]
            step_name = self.profile_data.steps_names[i]
            row = [step_name]
            if show_gpu:
                row.extend([costs.costs[ProfileRole.Kernel],
                            build_part_time_str(costs.costs[ProfileRole.Kernel], 'Kernel'),
                            costs.costs[ProfileRole.Memcpy],
                            build_part_time_str(costs.costs[ProfileRole.Memcpy], 'Memcpy'),
                            costs.costs[ProfileRole.Memset],
                            build_part_time_str(costs.costs[ProfileRole.Memset], 'Memset')])
            if self.profile_data.has_communication:
                row.extend([costs.costs[ProfileRole.Communication],
                            build_part_time_str(costs.costs[ProfileRole.Communication], 'Communication')])
            if show_gpu:
                row.extend([costs.costs[ProfileRole.Runtime],
                            build_part_time_str(costs.costs[ProfileRole.Runtime], 'Runtime')])
            row.extend([costs.costs[ProfileRole.DataLoader],
                        build_part_time_str(costs.costs[ProfileRole.DataLoader], 'DataLoader'),
                        costs.costs[ProfileRole.CpuOp],
                        build_part_time_str(costs.costs[ProfileRole.CpuOp], 'CPU Exec'),
                        costs.costs[ProfileRole.Other],
                        build_part_time_str(costs.costs[ProfileRole.Other], 'Other')])
            data['steps']['rows'].append(row)

        avg_costs = []
        if show_gpu:
            avg_costs.extend([
                build_avg_cost_dict('Kernel', self.profile_data.avg_costs.costs[ProfileRole.Kernel]),
                build_avg_cost_dict('Memcpy', self.profile_data.avg_costs.costs[ProfileRole.Memcpy]),
                build_avg_cost_dict('Memset', self.profile_data.avg_costs.costs[ProfileRole.Memset])
            ])
        if self.profile_data.has_communication:
            avg_costs.extend([
                build_avg_cost_dict('Communication', self.profile_data.avg_costs.costs[ProfileRole.Communication])
            ])
        if show_gpu:
            avg_costs.extend([
                build_avg_cost_dict('Runtime', self.profile_data.avg_costs.costs[ProfileRole.Runtime])
            ])
        avg_costs.extend([
            build_avg_cost_dict('DataLoader', self.profile_data.avg_costs.costs[ProfileRole.DataLoader]),
            build_avg_cost_dict('CPU Exec', self.profile_data.avg_costs.costs[ProfileRole.CpuOp]),
            build_avg_cost_dict('Other', self.profile_data.avg_costs.costs[ProfileRole.Other])
        ])

        data['performance'] = [{'name': 'Average Step Time', 'description': '',
                                'value': round(self.profile_data.avg_costs.costs[ProfileRole.Total]),
                                'extra': 100, 'children': avg_costs}]

        if len(self.profile_data.recommendations) == 0:
            html = '<li>N/A</li>'
        else:
            html = ''
            for recommendation in self.profile_data.recommendations:
                html += '<li>{}</li>'.format(recommendation)
        data['recommendations'] = '<ul>{}</ul>'.format(html)

        return data

    def _generate_op_pie(self, group_by_input_shape: bool = False):
        op_device_total_time = []
        op_device_self_time = []
        op_host_total_time = []
        op_host_self_time = []

        if group_by_input_shape:
            op_list = self.profile_data.op_list_groupby_name_input
        else:
            op_list = self.profile_data.op_list_groupby_name

        for op_agg in op_list:
            # Whether device_duration & self_device_duration are accurate or not depends on the input tracing data.
            if op_agg.device_duration > 0:
                op_device_total_time.append([op_agg.name, op_agg.device_duration])
            if op_agg.self_device_duration > 0:
                op_device_self_time.append([op_agg.name, op_agg.self_device_duration])
            if op_agg.host_duration > 0:
                op_host_total_time.append([op_agg.name, op_agg.host_duration])
            if op_agg.self_host_duration > 0:
                op_host_self_time.append([op_agg.name, op_agg.self_host_duration])

        op_device_total_time.sort(key=lambda x: x[1], reverse=True)
        op_device_self_time.sort(key=lambda x: x[1], reverse=True)
        op_host_total_time.sort(key=lambda x: x[1], reverse=True)
        op_host_self_time.sort(key=lambda x: x[1], reverse=True)

        data = {}
        device_total_time = {}
        device_self_time = {}
        host_total_time = {}
        host_self_time = {}

        if len(op_device_total_time) > 0:
            device_total_time['title'] = 'Device Total Time (us)'
            device_total_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            device_total_time['rows'] = op_device_total_time
        else:
            device_total_time = None

        if len(op_device_self_time) > 0:
            device_self_time['title'] = 'Device Self Time (us)'
            device_self_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            device_self_time['rows'] = op_device_self_time
        else:
            device_self_time = None

        if len(op_host_total_time) > 0:
            host_total_time['title'] = 'Host Total Time (us)'
            host_total_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            host_total_time['rows'] = op_host_total_time
        else:
            host_total_time = None

        if len(op_host_self_time) > 0:
            host_self_time['title'] = 'Host Self Time (us)'
            host_self_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            host_self_time['rows'] = op_host_self_time
        else:
            host_self_time = None

        data['device_total_time'] = device_total_time
        data['device_self_time'] = device_self_time
        data['host_total_time'] = host_total_time
        data['host_self_time'] = host_self_time

        return data

    def _generate_op_table(self, op_list: Iterable[OperatorAgg], group_by_input_shape=False, call_stack=False):
        show_gpu = self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset

        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        op_list = sorted(op_list,
                         key=lambda x: x.self_device_duration if show_gpu else x.self_host_duration,
                         reverse=True)

        data = list()
        result = {
            'metadata': {
                'sort': 'device_self_duration' if show_gpu else 'host_self_duration',
                'tooltips': {
                    'tc_eligible': consts.TOOLTIP_OP_TC_ELIGIBLE,
                    'tc_self_ratio': consts.TOOLTIP_OP_TC_SELF,
                    'tc_total_ratio': consts.TOOLTIP_OP_TC_TOTAL
                }
            },
            'data': data
        }
        for op in op_list:
            # Whether device_duration & self_device_duration are accurate or not depends on the input tracing data.
            row = dict()
            row['name'] = op.name
            if group_by_input_shape:
                row['input_shape'] = op.input_shape
            row['calls'] = op.calls
            if show_gpu:
                row['device_self_duration'] = round(op.self_device_duration)
                row['device_total_duration'] = round(op.device_duration)
            row['host_self_duration'] = round(op.self_host_duration)
            row['host_total_duration'] = round(op.host_duration)
            row['tc_eligible'] = 'Yes' if op.tc_eligible else 'No'
            row['tc_self_ratio'] = round(100 * op.tc_self_ratio, 2)
            row['tc_total_ratio'] = round(100 * op.tc_total_ratio, 2)
            if call_stack:
                row['call_stack'] = op.callstacks.pop()
            else:
                if group_by_input_shape:
                    key = op.name + '###' + str(op.input_shape)
                else:
                    key = op.name
                row['has_call_stack'] = key in stack_list_dict
            data.append(row)

        return result

    def _generate_op_table_for_stack(self, group_by_input_shape: bool):
        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        result = dict()
        for k, v in stack_list_dict.items():
            result[k] = self._generate_op_table(v, group_by_input_shape, True)
        return result

    def _generate_kernel_op_table(self):
        table = {}
        result = {
            'metadata': {
                'sort': 'Total Duration (us)'
            },
            'data': table
        }
        table['columns'] = [{'type': 'string', 'name': 'Name'},
                            {'type': 'string', 'name': 'Operator'},
                            {'type': 'string', 'name': 'Grid'},
                            {'type': 'string', 'name': 'Block'},
                            {'type': 'number', 'name': 'Register Per Thread'},
                            {'type': 'number', 'name': 'Shared Memory'},
                            {'type': 'string', 'name': 'Kernel Uses Tensor Cores',
                             'tooltip': consts.TOOLTIP_KERNEL_USES_TC},
                            {'type': 'string', 'name': 'Op is Tensor Cores eligible',
                             'tooltip': consts.TOOLTIP_KERNEL_OP_TC_ELIGIBLE}]
        col_names = ['Calls', 'Total Duration (us)', 'Mean Duration (us)', 'Max Duration (us)', 'Min Duration (us)']
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        gpu_metrics_columns = self.profile_data.gpu_metrics_parser.get_gpu_metrics_columns()
        table['columns'].extend(gpu_metrics_columns)

        table['rows'] = []
        kernel_list: List[KernelAggByNameOp] = sorted(
            self.profile_data.kernel_list_groupby_name_op, key=lambda x: x.total_duration, reverse=True)
        for agg_by_name_op in kernel_list:
            kernel_op_row = [agg_by_name_op.name, agg_by_name_op.op_name,
                             str(agg_by_name_op.grid), str(agg_by_name_op.block),
                             str(agg_by_name_op.regs_per_thread or '0'), str(agg_by_name_op.shared_memory or '0'),
                             'Yes' if agg_by_name_op.tc_used else 'No',
                             'Yes' if agg_by_name_op.op_tc_eligible else 'No',
                             agg_by_name_op.calls,
                             agg_by_name_op.total_duration, round(agg_by_name_op.avg_duration),
                             agg_by_name_op.max_duration, agg_by_name_op.min_duration]
            if self.profile_data.gpu_metrics_parser.has_blocks_per_sm:
                kernel_op_row.append(round(agg_by_name_op.avg_blocks_per_sm, 2))
            if self.profile_data.gpu_metrics_parser.has_occupancy:
                kernel_op_row.append(round(agg_by_name_op.avg_occupancy, 2))
            table['rows'].append(kernel_op_row)
        return result

    def _generate_kernel_pie(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            pie['rows'].append([name, row['sum']])
        data = {'total': pie}
        return data

    def _generate_kernel_table(self):
        table = {}
        result = {
            'metadata': {
                'sort': 'Total Duration (us)'
            },
            'data': table
        }
        table['columns'] = [{'type': 'string', 'name': 'Name'},
                            {'type': 'string', 'name': 'Tensor Cores Used',
                             'tooltip': consts.TOOLTIP_KERNEL_USES_TC}]
        columns = ['count', 'sum', 'mean', 'max', 'min']
        round_digits = [0, 0, 0, 0, 0]
        if self.profile_data.gpu_metrics_parser.has_blocks_per_sm:
            columns.append('blocks_per_sm')
            round_digits.append(2)
        if self.profile_data.gpu_metrics_parser.has_occupancy:
            columns.append('occupancy')
            round_digits.append(2)
        col_names = ['Calls', 'Total Duration (us)', 'Mean Duration (us)', 'Max Duration (us)', 'Min Duration (us)']
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        gpu_metrics_columns = self.profile_data.gpu_metrics_parser.get_gpu_metrics_columns()
        table['columns'].extend(gpu_metrics_columns)

        table['rows'] = []
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            kernel_row = [name, 'Yes' if row['tc_used'] else 'No']
            for i, column in enumerate(columns):
                kernel_row.append(round(row[column]) if round_digits[i] == 0
                                  else round(row[column], round_digits[i]))
            table['rows'].append(kernel_row)
        return result

    def _generate_tc_pie(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        pie['rows'].append(['Using Tensor Cores', self.profile_data.tc_used_ratio])
        pie['rows'].append(['Not Using Tensor Cores', 1.0 - self.profile_data.tc_used_ratio])
        data = {'total': pie}
        return data

    @staticmethod
    def _get_gpu_info(device_props, gpu_id):
        if (device_props is None) or (gpu_id >= len(device_props)) or (gpu_id < 0):
            return None

        device_prop: Dict = device_props[gpu_id]
        gpu_info = {}
        name = device_prop.get('name')
        if name is not None:
            gpu_info['Name'] = name

        mem = device_prop.get('totalGlobalMem')
        if mem is not None:
            gpu_info['Memory'] = '{} GB'.format(round(float(mem) / 1024 / 1024 / 1024, 2))
            gpu_info['Memory Raw'] = mem

        major = device_prop.get('computeMajor')
        minor = device_prop.get('computeMinor')
        if major is not None and minor is not None:
            gpu_info['Compute Capability'] = '{}.{}'.format(major, minor)

        return gpu_info


class DistributedRunGenerator:
    def __init__(self, all_profile_data: Iterable[DistributedRunProfileData], span):
        self.all_profile_data = all_profile_data
        self.span = span

    def generate_run_profile(self):
        profile_run = DistributedRunProfile(self.span)
        profile_run.views.append(consts.DISTRIBUTED_VIEW)
        profile_run.gpu_info = self._generate_gpu_info()
        profile_run.steps_to_overlap = self._generate_overlap_graph()
        profile_run.steps_to_wait = self._generate_wait_graph()
        profile_run.comm_ops = self._generate_ops_table()
        return profile_run

    def _generate_gpu_info(self):
        # first key is node name, the second key is process id, the third key is GPU0/,
        # the value is the gpu info json
        result: Dict[str, Dict[str, Dict[str, Dict]]] = OrderedDict()
        index = 0
        for data in sorted(self.all_profile_data, key=lambda x: x.worker):
            if not data.device_props:
                continue

            match = consts.NODE_PROCESS_PATTERN.match(data.worker)
            if match:
                node = match.group(1)
                process_id = match.group(2)
            else:
                logger.warning('cannot parse node name from worker name {}'.format(data.worker))
                node = data.worker
                process_id = index
                index += 1
            if node not in result:
                result[node] = OrderedDict()

            process_id = 'Process ' + str(process_id)
            result[node][process_id] = OrderedDict()
            for used_device in data.used_devices:
                gpu_info = RunGenerator._get_gpu_info(data.device_props, used_device)
                if gpu_info is not None:
                    result[node][process_id]['GPU'+str(used_device)] = gpu_info

        if result:
            for k, v in result.items():
                result[k] = OrderedDict(sorted(v.items()))
            return {
                'metadata': {'title': 'Device Information'},
                'data': result
            }
        else:
            return None

    def _generate_overlap_graph(self):
        result = dict()
        result['metadata'] = {
            'title': 'Computation/Communication Overview',
            'legends': ['Computation', 'Overlapping', 'Communication', 'Other'],
            'units': 'us'
        }
        steps_to_overlap: Dict[str, Dict[str, List[int]]] = OrderedDict()
        steps_to_overlap['all'] = OrderedDict()
        for data in self.all_profile_data:
            steps_to_overlap['all'][data.worker] = [0, 0, 0, 0]
            step_number = len(data.steps_names)
            for i, step_name in enumerate(data.steps_names):
                steps_to_overlap.setdefault(step_name, OrderedDict())
                costs = data.comm_overlap_costs[i]
                steps_to_overlap[step_name][data.worker] = [
                    costs.computation - costs.overlap,
                    costs.overlap,
                    costs.communication - costs.overlap,
                    costs.other
                ]
                steps_to_overlap['all'][data.worker] = [
                    sum(x) for x in zip(steps_to_overlap['all'][data.worker], steps_to_overlap[step_name][data.worker])]
            steps_to_overlap['all'][data.worker] = [int(x / max(1, step_number)) for x in
                                                    steps_to_overlap['all'][data.worker]]
        for k, v in steps_to_overlap.items():
            steps_to_overlap[k] = OrderedDict(sorted(v.items()))
        result['data'] = steps_to_overlap
        return result

    def _generate_wait_graph(self):
        result = dict()
        result['metadata'] = {
            'title': 'Synchronizing/Communication Overview',
            'legends': ['Data Transfer Time', 'Synchronizing Time'],
            'units': 'us'
        }
        steps_to_wait: Dict[str, Dict[str, List[int]]] = OrderedDict()

        steps_to_wait['all'] = OrderedDict()
        for data in self.all_profile_data:
            steps_to_wait['all'][data.worker] = [0, 0]
            step_number = len(data.step_comm_stats.values())
            for step, comm_stats in data.step_comm_stats.items():
                steps_to_wait.setdefault(step, OrderedDict())[data.worker] = [
                    comm_stats[1],
                    comm_stats[0]-comm_stats[1]
                ]
                steps_to_wait['all'][data.worker] = [
                    sum(x) for x in zip(steps_to_wait['all'][data.worker], steps_to_wait[step][data.worker])]
            steps_to_wait['all'][data.worker] = [int(x / max(1, step_number)) for x in
                                                 steps_to_wait['all'][data.worker]]

        for k, v in steps_to_wait.items():
            steps_to_wait[k] = OrderedDict(sorted(v.items()))
        result['data'] = steps_to_wait
        return result

    def _generate_ops_table(self):
        result = dict()
        result['metadata'] = {'title': 'Communication Operations Stats'}
        workers_to_comm_ops = OrderedDict()
        # Ignore the span for distributed view
        for data in self.all_profile_data:
            table = {}
            table['columns'] = [{'type': 'string', 'name': 'Name'}]
            col_names = [
                'Calls',
                'Total Size (bytes)',
                'Avg Size (bytes)',
                'Total Latency (us)',
                'Avg Latency (us)',
                'Data Transfer Time (us)',
                'Avg Data Transfer Time (us)'
            ]
            for column in col_names:
                table['columns'].append({'type': 'number', 'name': column})
            table['rows'] = []
            for op, stats in data.total_comm_stats.items():
                row = [
                    op,
                    stats[0],
                    stats[1],
                    round(stats[1]/stats[0]),
                    stats[2],
                    round(stats[2]/stats[0]),
                    stats[3],
                    round(stats[3]/stats[0])
                ]
                table['rows'].append(row)
            workers_to_comm_ops[data.worker] = table
        result['data'] = OrderedDict(sorted(workers_to_comm_ops.items()))
        return result
