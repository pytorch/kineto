# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import consts
from ..run import RunProfile


class RunGenerator(object):
    def __init__(self, worker, profile_data):
        self.worker = worker
        self.profile_data = profile_data

    def generate_run_profile(self):
        profile_run = RunProfile(self.worker)
        profile_run.is_gpu_used = self.profile_data.is_gpu_used
        profile_run.views.append(consts.OVERALL_VIEW)
        profile_run.overview = self._generate_overview()

        profile_run.views.append(consts.OP_VIEW)
        profile_run.operation_pie_by_name = self._generate_op_pie()
        profile_run.operation_table_by_name = self._generate_op_table()
        profile_run.operation_pie_by_name_input = self._generate_op_pie(True)
        profile_run.operation_table_by_name_input = self._generate_op_table(True)

        if self.profile_data.is_gpu_used:
            profile_run.views.append(consts.KERNEL_VIEW)
            profile_run.kernel_op_table = self._generate_kernel_op_table()
            profile_run.kernel_pie = self._generate_kernel_pie()
            profile_run.kernel_table = self._generate_kernel_table()

        profile_run.views.append(consts.TRACE_VIEW)
        profile_run.trace_file_path = self.profile_data.trace_file_path

        return profile_run

    def _generate_overview(self):
        show_gpu = self.profile_data.is_gpu_used

        data = {}
        data["steps"] = {}
        data["steps"]["columns"] = [{"type": "string", "name": "Step"}]
        if show_gpu:
            data["steps"]["columns"].extend([{"type": "number", "name": "Kernel"},
                                             {"type": "number", "name": "Memcpy"},
                                             {"type": "number", "name": "Memset"},
                                             {"type": "number", "name": "Runtime"}])
        data["steps"]["columns"].extend([{"type": "number", "name": "DataLoader"},
                                         {"type": "number", "name": "CPU Exec"},
                                         {"type": "number", "name": "Other"}])

        data["steps"]["rows"] = []
        for i in range(len(self.profile_data.steps_costs)):
            costs = self.profile_data.steps_costs[i]
            row = [self.profile_data.steps_names[i]]
            if show_gpu:
                row.extend([costs.kernel_cost, costs.memcpy_cost, costs.memset_cost, costs.runtime_cost])
            row.extend([costs.dataloader_cost, costs.cpuop_cost, costs.other_cost])
            data["steps"]["rows"].append(row)

        avg_costs = []
        if show_gpu:
            avg_costs.extend([
                {"name": "Kernel", "description": "",
                 "value": str(round(self.profile_data.avg_costs.kernel_cost)) + " us",
                 "extra": str(
                     round(100 * self.profile_data.avg_costs.kernel_cost / self.profile_data.avg_costs.step_total_cost,
                           2)) + "%"},
                {"name": "Memcpy", "description": "",
                 "value": str(round(self.profile_data.avg_costs.memcpy_cost)) + " us",
                 "extra": str(
                     round(100 * self.profile_data.avg_costs.memcpy_cost / self.profile_data.avg_costs.step_total_cost,
                           2)) + "%"},
                {"name": "Memset", "description": "",
                 "value": str(round(self.profile_data.avg_costs.memset_cost)) + " us",
                 "extra": str(
                     round(100 * self.profile_data.avg_costs.memset_cost / self.profile_data.avg_costs.step_total_cost,
                           2)) + "%"},
                {"name": "Runtime", "description": "",
                 "value": str(round(self.profile_data.avg_costs.runtime_cost)) + " us",
                 "extra": str(
                     round(100 * self.profile_data.avg_costs.runtime_cost / self.profile_data.avg_costs.step_total_cost,
                           2)) + "%"}])
        avg_costs.extend([
            {"name": "DataLoader", "description": "",
             "value": str(round(self.profile_data.avg_costs.dataloader_cost)) + " us",
             "extra": str(
                 round(100 * self.profile_data.avg_costs.dataloader_cost / self.profile_data.avg_costs.step_total_cost,
                       2)) + "%"},
            {"name": "CPU Exec", "description": "",
             "value": str(round(self.profile_data.avg_costs.cpuop_cost)) + " us",
             "extra": str(
                 round(100 * self.profile_data.avg_costs.cpuop_cost / self.profile_data.avg_costs.step_total_cost,
                       2)) + "%"},
            {"name": "Other", "description": "",
             "value": str(round(self.profile_data.avg_costs.other_cost)) + " us",
             "extra": str(
                 round(100 * self.profile_data.avg_costs.other_cost / self.profile_data.avg_costs.step_total_cost,
                       2)) + "%"}])

        data["performance"] = [{"name": "Average Step Time", "description": "",
                                "value": str(round(self.profile_data.avg_costs.step_total_cost)) + " us",
                                "extra": "100%", "children": avg_costs}]

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
            device_total_time["title"] = "Device Total Time"
            device_total_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            device_total_time["rows"] = op_device_total_time
        else:
            device_total_time = None

        if len(op_device_self_time) > 0:
            device_self_time["title"] = "Device Self Time"
            device_self_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            device_self_time["rows"] = op_device_self_time
        else:
            device_self_time = None

        if len(op_host_total_time) > 0:
            host_total_time["title"] = "Host Total Time"
            host_total_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            host_total_time["rows"] = op_host_total_time
        else:
            host_total_time = None

        if len(op_host_self_time) > 0:
            host_self_time["title"] = "Host Self Time"
            host_self_time["columns"] = [{"type": "string", "name": "name"}, {"type": "number", "name": "value"}]
            host_self_time["rows"] = op_host_self_time
        else:
            host_self_time = None

        data["device_total_time"] = device_total_time
        data["device_self_time"] = device_self_time
        data["host_total_time"] = host_total_time
        data["host_self_time"] = host_self_time

        return data

    def _generate_op_table(self, group_by_input_shape=False):
        show_gpu = self.profile_data.is_gpu_used

        columns = [{"type": "string", "name": "Name"}]
        if group_by_input_shape:
            columns.append({"type": "string", "name": "Input Shape"})

        columns.append({"type": "number", "name": "Calls"})
        if show_gpu:
            columns.extend([{"type": "number", "name": "Device Self Duration (us)"},
                            {"type": "number", "name": "Device Total Duration (us)"}])

        columns.extend([{"type": "number", "name": "Host Self Duration (us)"},
                        {"type": "number", "name": "Host Total Duration (us)"}])

        if group_by_input_shape:
            op_list = self.profile_data.op_list_groupby_name_input
        else:
            op_list = self.profile_data.op_list_groupby_name

        op_list = sorted(op_list,
                         key=lambda x: x.self_device_duration if show_gpu else x.self_host_duration,
                         reverse=True)

        rows = []
        for op in op_list:
            # Whether device_duration & self_device_duration are accurate or not depends on the input tracing data.
            row = [op.name]
            if group_by_input_shape:
                row.append(op.input_shape)

            row.append(op.calls)
            if show_gpu:
                row.extend([round(op.self_device_duration), round(op.device_duration)])

            row.extend([round(op.self_host_duration), round(op.host_duration)])
            rows.append(row)

        data = {"data": {"columns": columns, "rows": rows}}
        return data

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
        for id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
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
        for id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            kernel_row = [name]
            for column in columns:
                kernel_row.append(round(row[column]))
            table["rows"].append(kernel_row)
        data = {"data": table}
        return data
