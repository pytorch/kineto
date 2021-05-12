# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from collections import OrderedDict
from .. import consts, utils
from ..run import RunProfile, AllRunProfile
from .overall_parser import ProfileRole


class RunGenerator(object):
    def __init__(self, worker, profile_data):
        self.worker = worker
        self.profile_data = profile_data

    def generate_run_profile(self):
        profile_run = RunProfile(self.worker)
        profile_run.has_runtime = self.profile_data.has_runtime
        profile_run.has_kernel = self.profile_data.has_kernel
        profile_run.has_communication = self.profile_data.has_communication
        profile_run.has_memcpy_or_memset = self.profile_data.has_memcpy_or_memset
        profile_run.views.append(consts.OVERALL_VIEW)
        profile_run.overview = self._generate_overview()

        profile_run.views.append(consts.OP_VIEW)
        profile_run.operation_pie_by_name = self._generate_op_pie()
        profile_run.operation_table_by_name = self._generate_op_table(self.profile_data.op_list_groupby_name)
        profile_run.operation_stack_by_name = self._generate_op_table_for_stack(False)
        profile_run.operation_pie_by_name_input = self._generate_op_pie(True)
        profile_run.operation_table_by_name_input = self._generate_op_table(self.profile_data.op_list_groupby_name_input, True)
        profile_run.operation_stack_by_name_input = self._generate_op_table_for_stack(True)

        if self.profile_data.has_kernel:
            profile_run.views.append(consts.KERNEL_VIEW)
            profile_run.kernel_op_table = self._generate_kernel_op_table()
            profile_run.kernel_pie = self._generate_kernel_pie()
            profile_run.kernel_table = self._generate_kernel_table()

        profile_run.views.append(consts.TRACE_VIEW)
        profile_run.trace_file_path = self.profile_data.trace_file_path

        return profile_run

    def _generate_overview(self):
        def build_part_time_str(part_cost, part_name):
            format_str = '<div class="visualization-tooltip" style="white-space: nowrap;">' \
                         'Step {}<br>' \
                         'Total: {}us<br>' \
                         '<b>{}: {}us</b><br>' \
                         'Percentage: {}%' \
                         '</div>'
            percentage = round(100 * part_cost / costs.costs[ProfileRole.Total], 2)
            return format_str.format(step_name, costs.costs[ProfileRole.Total], part_name, part_cost, percentage)

        def build_avg_cost_dict(part_name, part_cost):
            cost_dict = {"name": part_name,
                         "description": "",
                         "value": round(part_cost),
                         "extra": round(100 * part_cost / self.profile_data.avg_costs.costs[ProfileRole.Total], 2)}
            return cost_dict

        show_gpu = self.profile_data.has_runtime or self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset

        column_tootip = {"type": "string", "role": "tooltip", "p": {"html": "true"}}
        data = {}
        data["steps"] = {}
        data["steps"]["columns"] = [{"type": "string", "name": "Step"}]
        if show_gpu:
            data["steps"]["columns"].extend([{"type": "number", "name": "Kernel"},
                                             column_tootip,
                                             {"type": "number", "name": "Memcpy"},
                                             column_tootip,
                                             {"type": "number", "name": "Communication"},
                                             column_tootip,
                                             {"type": "number", "name": "Memset"},
                                             column_tootip,
                                             {"type": "number", "name": "Runtime"},
                                             column_tootip])
        data["steps"]["columns"].extend([{"type": "number", "name": "DataLoader"},
                                         column_tootip,
                                         {"type": "number", "name": "CPU Exec"},
                                         column_tootip,
                                         {"type": "number", "name": "Other"},
                                         column_tootip])

        data["steps"]["rows"] = []
        for i in range(len(self.profile_data.steps_costs)):
            costs = self.profile_data.steps_costs[i]
            step_name = self.profile_data.steps_names[i]
            row = [step_name]
            if show_gpu:
                row.extend([costs.costs[ProfileRole.Kernel],
                            build_part_time_str(costs.costs[ProfileRole.Kernel], "Kernel"),
                            costs.costs[ProfileRole.Memcpy],
                            build_part_time_str(costs.costs[ProfileRole.Memcpy], "Memcpy"),
                            costs.costs[ProfileRole.Memset],
                            build_part_time_str(costs.costs[ProfileRole.Memset], "Memset"),
                            costs.costs[ProfileRole.Communication],
                            build_part_time_str(costs.costs[ProfileRole.Communication], "Communication"),
                            costs.costs[ProfileRole.Runtime],
                            build_part_time_str(costs.costs[ProfileRole.Runtime], "Runtime")])
            row.extend([costs.costs[ProfileRole.DataLoader],
                        build_part_time_str(costs.costs[ProfileRole.DataLoader], "DataLoader"),
                        costs.costs[ProfileRole.CpuOp],
                        build_part_time_str(costs.costs[ProfileRole.CpuOp], "CPU Exec"),
                        costs.costs[ProfileRole.Other],
                        build_part_time_str(costs.costs[ProfileRole.Other], "Other")])
            data["steps"]["rows"].append(row)

        avg_costs = []
        if show_gpu:
            avg_costs.extend([
                build_avg_cost_dict("Kernel", self.profile_data.avg_costs.costs[ProfileRole.Kernel]),
                build_avg_cost_dict("Memcpy", self.profile_data.avg_costs.costs[ProfileRole.Memcpy]),
                build_avg_cost_dict("Memset", self.profile_data.avg_costs.costs[ProfileRole.Memset]),
                build_avg_cost_dict("Communication", self.profile_data.avg_costs.costs[ProfileRole.Communication]),
                build_avg_cost_dict("Runtime", self.profile_data.avg_costs.costs[ProfileRole.Runtime])
            ])
        avg_costs.extend([
            build_avg_cost_dict("DataLoader", self.profile_data.avg_costs.costs[ProfileRole.DataLoader]),
            build_avg_cost_dict("CPU Exec", self.profile_data.avg_costs.costs[ProfileRole.CpuOp]),
            build_avg_cost_dict("Other", self.profile_data.avg_costs.costs[ProfileRole.Other])
        ])

        data["performance"] = [{"name": "Average Step Time", "description": "",
                                "value": round(self.profile_data.avg_costs.costs[ProfileRole.Total]),
                                "extra": 100, "children": avg_costs}]

        if len(self.profile_data.recommendations) == 0:
            html = "<li>N/A</li>"
        else:
            html = ""
            for recommendation in self.profile_data.recommendations:
                html += "<li>{}</li>".format(recommendation)
        data["recommendations"] = "<ul>{}</ul>".format(html)

        return data

    def _generate_op_pie(self, group_by_input_shape=False):
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
            device_total_time["title"] = "Device Total Time (us)"
            device_total_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            device_total_time["rows"] = op_device_total_time
        else:
            device_total_time = None

        if len(op_device_self_time) > 0:
            device_self_time["title"] = "Device Self Time (us)"
            device_self_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            device_self_time["rows"] = op_device_self_time
        else:
            device_self_time = None

        if len(op_host_total_time) > 0:
            host_total_time["title"] = "Host Total Time (us)"
            host_total_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            host_total_time["rows"] = op_host_total_time
        else:
            host_total_time = None

        if len(op_host_self_time) > 0:
            host_self_time["title"] = "Host Self Time (us)"
            host_self_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            host_self_time["rows"] = op_host_self_time
        else:
            host_self_time = None

        data["device_total_time"] = device_total_time
        data["device_self_time"] = device_self_time
        data["host_total_time"] = host_total_time
        data["host_self_time"] = host_self_time

        return data

    def _generate_op_table(self, op_list, group_by_input_shape=False, call_stack=False):
        show_gpu = self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset

        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        op_list = sorted(op_list,
                         key=lambda x: x.self_device_duration if show_gpu else x.self_host_duration,
                         reverse=True)

        data = list()
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
            if call_stack:
                row['call_stack'] = op.call_stacks.pop()
            else:
                if group_by_input_shape:
                    key = op.name + '###' + str(op.input_shape)
                else:
                    key = op.name
                row['has_call_stack'] = key in stack_list_dict
            data.append(row)

        return data

    def _generate_op_table_for_stack(self, group_by_input_shape):
        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        result = dict()
        for k,v in stack_list_dict.items():
            result[k] = self._generate_op_table(v, group_by_input_shape, True)
        return result

    def _generate_kernel_op_table(self):
        table = {}
        table["columns"] = [{"type": "string", "name": "Name"}, {"type": "string", "name": "Operator"}]
        col_names = ["Calls", "Total Duration (us)", "Mean Duration (us)", "Max Duration (us)", "Min Duration (us)"]
        for column in col_names:
            table["columns"].append({"type": "number", "name": column})
        table["rows"] = []
        kernel_list = sorted(self.profile_data.kernel_list_groupby_name_op, key=lambda x: x.total_duration,
                             reverse=True)
        for agg_by_name_op in kernel_list:
            kernel_op_row = [agg_by_name_op.name, agg_by_name_op.op_name, agg_by_name_op.calls,
                             agg_by_name_op.total_duration, agg_by_name_op.avg_duration,
                             agg_by_name_op.min_duration, agg_by_name_op.max_duration]
            table["rows"].append(kernel_op_row)
        data = {"data": table}
        return data

    def _generate_kernel_pie(self):
        pie = {"columns": [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}], "rows": []}
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            pie["rows"].append([name, row["sum"]])
        data = {"total": pie}
        return data

    def _generate_kernel_table(self):
        table = {}
        table["columns"] = [{"type": "string", "name": "Name"}]
        columns = ["count", "sum", "mean", "max", "min"]
        col_names = ["Calls", "Total Duration (us)", "Mean Duration (us)", "Max Duration (us)", "Min Duration (us)"]
        for column in col_names:
            table["columns"].append({"type": "number", "name": column})
        table["rows"] = []
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            kernel_row = [name]
            for column in columns:
                kernel_row.append(round(row[column]))
            table["rows"].append(kernel_row)
        data = {"data": table}
        return data

class AllRunGenerator(object):
    def __init__(self, all_profile_data):
        self.all_profile_data = all_profile_data

    def generate_all_run_profile(self):
        all_profile_run = AllRunProfile()
        all_profile_run.views.append(consts.DISTRIBUTED_VIEW)
        all_profile_run.steps_to_overlap = self._generate_overlap_graph()
        all_profile_run.steps_to_wait = self._generate_wait_graph()
        all_profile_run.comm_ops = self._generate_ops_table()
        return all_profile_run

    def _generate_overlap_graph(self):
        steps_to_overlap = OrderedDict()
        for worker,data in self.all_profile_data.items():
            for i in range(len(data.steps_names)):
                step_name = data.steps_names[i]
                if step_name not in steps_to_overlap:
                    steps_to_overlap[step_name] = OrderedDict()
                costs = data.comm_overlap_costs[i]
                steps_to_overlap[step_name][worker] = [costs.computation - costs.overlap, costs.overlap, costs.communication - costs.overlap, costs.other]
        return steps_to_overlap

    def _generate_wait_graph(self):
        steps_to_wait = OrderedDict()
        for worker,data in self.all_profile_data.items():
            for step,comm_stats in data.step_comm_stats.items():
                if step not in steps_to_wait:
                    steps_to_wait[step] = OrderedDict()
                steps_to_wait[step][worker] = [comm_stats[0]-comm_stats[1], comm_stats[1]]
        return steps_to_wait

    def _generate_ops_table(self):
        workers_to_comm_ops = OrderedDict()
        for worker,data in self.all_profile_data.items():
            table = {}
            table["columns"] = [{"type": "string", "name": "Name"}]
            col_names = ["Calls", "Total Size (bytes)", "Total Latency (us)", "Avg Latency (us)", "Real Time (us)", "Avg Real time (us)"]
            for column in col_names:
                table["columns"].append({"type": "number", "name": column})
            table["rows"] = []
            for op,stats in data.total_comm_stats.items():
                row = [op, stats[0], stats[1], round(stats[1]/stats[0]), stats[2], round(stats[2]/stats[0]), stats[3], round(stats[3]/stats[0])]
                table["rows"].append(row)
            workers_to_comm_ops[worker] = table
        return workers_to_comm_ops
