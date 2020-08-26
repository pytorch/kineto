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

# Lint as: python3
"""Tests for kernel_stats_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import enum

import gviz_api
import unittest

from tensorboard_plugin_profile.convert import kernel_stats_proto_to_gviz
from tensorboard_plugin_profile.protobuf import kernel_stats_pb2


class StrEnum(str, enum.Enum):
  pass


class MockValues(StrEnum):
  RANK = 1
  KERNEL_NAME = "kernel name foo"
  REGISTERS_PER_THREAD = 32
  STATIC_SHMEM_BYTES = 32768
  DYNAMIC_SHMEM_BYTES = 32768
  BLOCK_DIM = "32,1,1"
  GRID_DIM = "80,1,1"
  IS_OP_TENSOR_CORE_ELIGIBLE = False
  IS_KERNEL_USING_TENSOR_CORE = False
  OP_NAME = "operation name bar"
  OCCURRENCES = 10
  TOTAL_DURATION_NS = 1000000
  AVG_DURATION_NS = 100000
  MIN_DURATION_NS = 10000
  MAX_DURATION_NS = 190000


class ProtoToGvizTest(unittest.TestCase):

  def create_empty_kernel_stats(self):
    return kernel_stats_pb2.KernelStatsDb()

  def create_mock_kernel_stats(self):
    kernel_stats = kernel_stats_pb2.KernelStatsDb()

    kernel_reports = []
    # Add 3 rows
    for _ in range(0, 3):
      kernel_report = kernel_stats_pb2.KernelReport()
      kernel_report.name = MockValues.KERNEL_NAME
      kernel_report.registers_per_thread = int(MockValues.REGISTERS_PER_THREAD)
      kernel_report.static_shmem_bytes = int(MockValues.STATIC_SHMEM_BYTES)
      kernel_report.dynamic_shmem_bytes = int(MockValues.DYNAMIC_SHMEM_BYTES)
      kernel_report.block_dim.extend(
          int(x) for x in MockValues.BLOCK_DIM.split(","))
      kernel_report.grid_dim.extend(
          int(x) for x in MockValues.GRID_DIM.split(","))
      kernel_report.is_op_tensor_core_eligible = \
          "True" == MockValues.IS_OP_TENSOR_CORE_ELIGIBLE
      kernel_report.is_kernel_using_tensor_core = \
          "True" == MockValues.IS_KERNEL_USING_TENSOR_CORE
      kernel_report.op_name = MockValues.OP_NAME
      kernel_report.occurrences = int(MockValues.OCCURRENCES)
      kernel_report.total_duration_ns = int(MockValues.TOTAL_DURATION_NS)
      kernel_report.min_duration_ns = int(MockValues.MIN_DURATION_NS)
      kernel_report.max_duration_ns = int(MockValues.MAX_DURATION_NS)
      kernel_reports.append(kernel_report)
    kernel_stats.reports.extend(kernel_reports)

    return kernel_stats

  def test_empty_kernel_stats(self):
    kernel_stats = self.create_empty_kernel_stats()
    data_table = kernel_stats_proto_to_gviz.generate_kernel_reports_table(
        kernel_stats.reports)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Kernel stats chart data table has 14 columns.
    self.assertEqual(len(data_table.columns), 14)

  def test_mock_kernel_stats(self):
    kernel_stats = self.create_mock_kernel_stats()
    (table_description, data, custom_properties
    ) = kernel_stats_proto_to_gviz.get_kernel_reports_table_args(
        kernel_stats.reports)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertEqual(len(data), 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Table descriptor is a list of 14 columns.
    self.assertEqual(len(table_description), 14)
    # DataTable also has 10 columns.
    self.assertEqual(len(data_table.columns), 14)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for (cc, column_header) in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        expected = [
            rr,
            MockValues.KERNEL_NAME,
            int(MockValues.REGISTERS_PER_THREAD),
            int(MockValues.STATIC_SHMEM_BYTES) +
            int(MockValues.DYNAMIC_SHMEM_BYTES),
            MockValues.BLOCK_DIM,
            MockValues.GRID_DIM,
            MockValues.IS_OP_TENSOR_CORE_ELIGIBLE,
            MockValues.IS_KERNEL_USING_TENSOR_CORE,
            MockValues.OP_NAME,
            int(MockValues.OCCURRENCES),
            int(MockValues.TOTAL_DURATION_NS) / 1000,
            int(MockValues.AVG_DURATION_NS) / 1000,
            int(MockValues.MIN_DURATION_NS) / 1000,
            int(MockValues.MAX_DURATION_NS) / 1000,
        ]

        for (cc, cell_str) in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]
          print(raw_value)
          print(str(expected[cc]))

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["boolean", "number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)

          # Check against expected values we have set in our mock table.
          if value_type == "boolean":
            # CSV converts boolean to lower case.
            self.assertEqual(str(expected_value).lower(), cell_str)
            self.assertEqual(expected[cc].lower(), cell_str)
          elif value_type == "string":
            self.assertEqual(expected[cc], cell_str)
          elif value_type == "number":
            self.assertEqual(str(expected_value), cell_str)
            if "." in cell_str:
              self.assertEqual(str(float(expected[cc])), cell_str)
            else:
              self.assertEqual(str(int(expected[cc])), cell_str)

if __name__ == '__main__':
    unittest.main()
