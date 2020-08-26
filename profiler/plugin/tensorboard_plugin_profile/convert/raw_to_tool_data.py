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
"""For conversion of raw files to tool data.

Usage:
    data = xspace_to_tool_data(xplane, tool, tqx)
    data = tool_proto_to_tool_data(tool_proto, tool, tqx)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tensorboard_plugin_profile.convert import input_pipeline_proto_to_gviz
from tensorboard_plugin_profile.convert import kernel_stats_proto_to_gviz
from tensorboard_plugin_profile.convert import overview_page_proto_to_gviz
from tensorboard_plugin_profile.convert import tf_stats_proto_to_gviz
from tensorboard_plugin_profile.convert import trace_events_json
from tensorboard_plugin_profile.protobuf import trace_events_pb2

logger = logging.getLogger('tensorboard')


def process_raw_trace(raw_trace):
  """Processes raw trace data and returns the UI data."""
  trace = trace_events_pb2.Trace()
  trace.ParseFromString(raw_trace)
  return ''.join(trace_events_json.TraceEventsJsonStream(trace))

def tool_proto_to_tool_data(tool_proto, tool, tqx):
  """Converts the serialized tool proto to tool data string.

  Args:
    tool_proto: A serialized XSpace proto string.
    tool: A string of tool name.
    tqx: Gviz output format.

  Returns:
    Returns a string of tool data.
  """
  data = ''
  if tool == 'trace_viewer':
    data = process_raw_trace(tool_proto)
  elif tool == 'tensorflow_stats':
    if tqx == 'out:csv;':
      data = tf_stats_proto_to_gviz.to_csv(tool_proto)
    else:
      data = tf_stats_proto_to_gviz.to_json(tool_proto)
  elif tool == 'overview_page@':
    data = overview_page_proto_to_gviz.to_json(tool_proto)
  elif tool == 'input_pipeline_analyzer@':
    data = input_pipeline_proto_to_gviz.to_json(tool_proto)
  elif tool == 'kernel_stats':
    if tqx == 'out:csv;':
      data = kernel_stats_proto_to_gviz.to_csv(tool_proto)
    else:
      data = kernel_stats_proto_to_gviz.to_json(tool_proto)
  else:
    logger.warning('%s is not a known tool', tool)
  return data
