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
"""Tests for input_pipeline_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tensorboard_plugin_profile.convert import diagnostics
from tensorboard_plugin_profile.protobuf import diagnostics_pb2


class DiagnosticsTest(unittest.TestCase):

  def test_error_simple(self):
    diag = diagnostics_pb2.Diagnostics()
    diag.info.append("info")
    diag.warnings.append("warning")
    diag.errors.append("error1")
    diag.errors.append("error2")
    diag_table = diagnostics.generate_diagnostics_table(diag)
    # There're two columns: severity and message.
    self.assertEqual(len(diag_table.columns), 2)
    self.assertEqual(4, diag_table.NumberOfRows(), "Error table has four rows.")

if __name__ == '__main__':
    unittest.main()
