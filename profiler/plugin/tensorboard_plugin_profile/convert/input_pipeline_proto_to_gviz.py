# Copyright 2020 Facebook, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For conversion of TB Input Pipeline Analyzer protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(ipa)
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.convert import diagnostics as diag
from tensorboard_plugin_profile.protobuf import input_pipeline_pb2


def get_step_breakdown_table_args(ipa):
  """Creates a step breakdown from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("stepnum", "string", "Step number"),
      ("deviceComputeTimeMs", "number", "Device compute"),
      ("deviceToDeviceTimeMs", "number", "Device to device"),
      ("hostComputeTimeMs", "number", "Host compute"),
      ("kernelLaunchTimeMs", "number", "Kernel launch"),
      ("infeedTimeMs", "number", "Input"),
      ("outfeedTimeMs", "number", "Output"),
      ("compileTimeMs", "number", "Compilation"),
      ("otherTimeMs", "number", "All others"),
      ("tooltip", "string", "tooltip", {
          "role": "tooltip"
      }),
  ]

  # Parameters for input analysis summary.
  total_step_time_ms = 0.0
  total_input_ms = 0.0
  total_output_ms = 0.0
  total_host_compute_ms = 0.0
  total_host_prepare_ms = 0.0
  total_host_compile_ms = 0.0
  total_device_to_device_ms = 0.0
  total_unknown_ms = 0.0

  data = []
  for step_details in ipa.step_details:
    details = input_pipeline_pb2.PerGenericStepDetails()
    step_details.Unpack(details)

    tooltip = ("Step {}, duration: {:.2f} ms\n"
               "-All others: {:.2f} ms\n"
               "-Compilation: {:.2f} ms\n"
               "-Output: {:.2f} ms\n"
               "-Input: {:.2f} ms\n"
               "-Kernel launch: {:.2f} ms\n"
               "-Host compute: {:.2f} ms\n"
               "-Device to device: {:.2f} ms\n"
               "-Device compute: {:.2f} ms").format(
                   details.step_number, details.step_time_ms,
                   details.unknown_time_ms, details.host_compile_ms,
                   details.output_ms,
                   details.host_wait_input_ms + details.host_to_device_ms,
                   details.host_prepare_ms, details.host_compute_ms,
                   details.device_to_device_ms, details.device_compute_ms)

    row = [
        str(details.step_number), details.device_compute_ms,
        details.device_to_device_ms, details.host_compute_ms,
        details.host_prepare_ms,
        details.host_wait_input_ms + details.host_to_device_ms,
        details.output_ms, details.host_compile_ms, details.unknown_time_ms,
        tooltip
    ]
    data.append(row)

    total_step_time_ms += details.step_time_ms
    total_input_ms += details.host_wait_input_ms + details.host_to_device_ms
    total_output_ms += details.output_ms
    total_host_prepare_ms += details.host_prepare_ms
    total_device_to_device_ms += details.device_to_device_ms
    total_host_compute_ms += details.host_compute_ms
    total_host_compile_ms += details.host_compile_ms
    total_unknown_ms += details.unknown_time_ms

  bottleneck_analysis = input_pipeline_pb2.BottleneckAnalysis()
  ipa.recommendation.bottleneck_analysis.Unpack(bottleneck_analysis)
  kernel_launch_classification = \
      bottleneck_analysis.kernel_launch_classification
  kernel_launch_statement = bottleneck_analysis.kernel_launch_statement
  all_other_classification = bottleneck_analysis.all_other_classification
  all_other_statement = bottleneck_analysis.all_other_statement
  input_conclusion = bottleneck_analysis.input_statement
  summary_next_step = ipa.recommendation.summary_next_step

  # Add step time summary
  steptime_ms_average = "{:.1f}".format(ipa.step_time_summary.average)
  steptime_ms_standard_deviation = "{:.1f}".format(
      ipa.step_time_summary.standard_deviation)
  steptime_ms_minimum = "{:.1f}".format(ipa.step_time_summary.minimum)
  steptime_ms_maximum = "{:.1f}".format(ipa.step_time_summary.maximum)

  # Add step time breakdown
  breakdown = input_pipeline_pb2.GenericStepTimeBreakdown()
  ipa.step_time_breakdown.Unpack(breakdown)
  device_compute_time_ms_avg = "{:.1f}".format(
      breakdown.device_compute_ms_summary.average)
  device_compute_time_ms_sdv = "{:.1f}".format(
      breakdown.device_compute_ms_summary.standard_deviation)
  device_to_device_time_ms_avg = "{:.1f}".format(
      breakdown.device_to_device_ms_summary.average)
  device_to_device_time_ms_sdv = "{:.1f}".format(
      breakdown.device_to_device_ms_summary.standard_deviation)
  infeed_time_ms_avg = "{:.1f}".format(breakdown.input_ms_summary.average)
  infeed_time_ms_sdv = "{:.1f}".format(
      breakdown.input_ms_summary.standard_deviation)
  outfeed_time_ms_avg = "{:.1f}".format(breakdown.output_ms_summary.average)
  outfeed_time_ms_sdv = "{:.1f}".format(
      breakdown.output_ms_summary.standard_deviation)
  host_compute_time_ms_avg = "{:.1f}".format(
      breakdown.host_compute_ms_summary.average)
  host_compute_time_ms_sdv = "{:.1f}".format(
      breakdown.host_compute_ms_summary.standard_deviation)
  kernel_launch_time_ms_avg = "{:.1f}".format(
      breakdown.host_prepare_ms_summary.average)
  kernel_launch_time_ms_sdv = "{:.1f}".format(
      breakdown.host_prepare_ms_summary.standard_deviation)
  compile_time_ms_avg = "{:.1f}".format(
      breakdown.host_compile_ms_summary.average)
  compile_time_ms_sdv = "{:.1f}".format(
      breakdown.host_compile_ms_summary.standard_deviation)
  other_time_ms_avg = "{:.1f}".format(breakdown.unknown_time_ms_summary.average)
  other_time_ms_sdv = "{:.1f}".format(
      breakdown.unknown_time_ms_summary.standard_deviation)

  custom_properties = {
      "hardware_type": ipa.hardware_type,
      # Step time summary
      "steptime_ms_average": steptime_ms_average,
      "steptime_ms_standard_deviation": steptime_ms_standard_deviation,
      "steptime_ms_minimum": steptime_ms_minimum,
      "steptime_ms_maximum": steptime_ms_maximum,
      # Step time breakdown
      "device_compute_time_ms_avg": device_compute_time_ms_avg,
      "device_compute_time_ms_sdv": device_compute_time_ms_sdv,
      "device_to_device_time_ms_avg": device_to_device_time_ms_avg,
      "device_to_device_time_ms_sdv": device_to_device_time_ms_sdv,
      "infeed_time_ms_avg": infeed_time_ms_avg,
      "infeed_time_ms_sdv": infeed_time_ms_sdv,
      "outfeed_time_ms_avg": outfeed_time_ms_avg,
      "outfeed_time_ms_sdv": outfeed_time_ms_sdv,
      "host_compute_time_ms_avg": host_compute_time_ms_avg,
      "host_compute_time_ms_sdv": host_compute_time_ms_sdv,
      "kernel_launch_time_ms_avg": kernel_launch_time_ms_avg,
      "kernel_launch_time_ms_sdv": kernel_launch_time_ms_sdv,
      "compile_time_ms_avg": compile_time_ms_avg,
      "compile_time_ms_sdv": compile_time_ms_sdv,
      "other_time_ms_avg": other_time_ms_avg,
      "other_time_ms_sdv": other_time_ms_sdv,
      # Input analysis summary
      "input_conclusion": input_conclusion,
      "summary_nextstep": summary_next_step,
      # Generic recommendation
      "kernel_launch_bottleneck": kernel_launch_classification,
      "kernel_launch_statement": kernel_launch_statement,
      "all_other_bottleneck": all_other_classification,
      "all_other_statement": all_other_statement,
  }

  return (table_description, data, custom_properties)


def get_input_op_table_args(ipa):
  """Creates an input operator from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("opName", "string", "Input Op"),
      ("count", "number", "Count"),
      ("timeInMs", "number", "Total Time (in ms)"),
      ("timeInPercent", "number",
       "Total Time (as % of total input-processing time)"),
      ("selfTimeInMs", "number", "Total Self Time (in ms)"),
      ("selfTimeInPercent", "number",
       "Total Self Time (as % of total input-processing time)"),
      ("category", "string", "Category"),
  ]

  data = []
  for details in ipa.input_op_details:
    row = [
        details.op_name,
        details.count,
        details.time_in_ms,
        details.time_in_percent / 100.0,
        details.self_time_in_ms,
        details.self_time_in_percent / 100.0,
        details.category,
    ]
    data.append(row)

  enqueue_us = "{:.3f}".format(ipa.input_time_breakdown.enqueue_us)
  demanded_file_read_us = "{:.3f}".format(
      ipa.input_time_breakdown.demanded_file_read_us)
  advanced_file_read_us = "{:.3f}".format(
      ipa.input_time_breakdown.advanced_file_read_us)
  preprocessing_us = "{:.3f}".format(ipa.input_time_breakdown.preprocessing_us)
  unclassified_non_enqueue_us = "{:.3f}".format(
      ipa.input_time_breakdown.unclassified_non_enqueue_us)

  custom_properties = {
      "enqueue_us": enqueue_us,
      "demanded_file_read_us": demanded_file_read_us,
      "advanced_file_read_us": advanced_file_read_us,
      "preprocessing_us": preprocessing_us,
      "unclassified_nonequeue_us": unclassified_non_enqueue_us,
  }

  return (table_description, data, custom_properties)


def get_recommendation_table_args(ipa):
  """Creates an recommendation table from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [("link", "string", "link")]

  data = [[detail] for detail in ipa.recommendation.details]

  return (table_description, data, None)


def generate_step_breakdown_table(ipa):
  (table_description, data,
   custom_properties) = get_step_breakdown_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_input_op_table(ipa):
  (table_description, data, custom_properties) = get_input_op_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_recommendation_table(ipa):
  (table_description, data,
   custom_properties) = get_recommendation_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(ipa):
  """Generates a list of gviz tables from InputPipelineAnalysisResult."""

  return [
      generate_step_breakdown_table(ipa),
      generate_input_op_table(ipa),
      generate_recommendation_table(ipa),
      diag.generate_diagnostics_table(ipa.diagnostics),
  ]


def to_json(raw_data):
  """Converts a serialized InputPipelineAnalysisResult string to json."""
  ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
  ipa.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(ipa)
  json_join = ",".join(x.ToJSon() for x in all_chart_tables)
  return "[" + json_join + "]"
