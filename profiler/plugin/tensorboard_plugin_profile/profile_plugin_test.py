# -*- coding: utf-8 -*-
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
''"Tests for the Profile plugin.''"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import tempfile
import unittest

from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard_plugin_profile import profile_plugin
from tensorboard_plugin_profile import profile_plugin_test_utils as utils
from tensorboard_plugin_profile.protobuf import trace_events_pb2
from torch.utils.tensorboard.writer import FileWriter

RUN_TO_TOOLS = {
    'foo': ['trace_viewer'],
    'bar': ['unsupported'],
    'baz': ['overview_page', 'op_profile', 'trace_viewer'],
    'qux': ['overview_page@', 'input_pipeline_analyzer@', 'trace_viewer'],
    'empty': [],
}

RUN_TO_HOSTS = {
    'foo': ['host0', 'host1'],
    'bar': ['host1'],
    'baz': ['host2'],
    'qux': [None],
    'empty': [],
}

EXPECTED_TRACE_DATA = dict(
    displayTimeUnit='ns',
    metadata={'highres-ticks': True},
    traceEvents=[
        dict(ph='M', pid=0, name='process_name', args=dict(name='foo')),
        dict(ph='M', pid=0, name='process_sort_index', args=dict(sort_index=0)),
        dict(),
    ],
)

# Suffix for the empty eventfile to write. Should be kept in sync with TF
# profiler kProfileEmptySuffix constant defined in:
#   tensorflow/core/profiler/rpc/client/capture_profile.cc.
EVENT_FILE_SUFFIX = '.profile-empty'


def generate_testdata(logdir):
  plugin_logdir = plugin_asset_util.PluginDirectory(
      logdir, profile_plugin.ProfilePlugin.plugin_name)
  os.makedirs(plugin_logdir)
  for run in RUN_TO_TOOLS:
    run_dir = os.path.join(plugin_logdir, run)
    os.mkdir(run_dir)
    for tool in RUN_TO_TOOLS[run]:
      if tool not in profile_plugin.TOOLS:
        continue
      for host in RUN_TO_HOSTS[run]:
        filename = profile_plugin._make_filename(host, tool)
        tool_file = os.path.join(run_dir, filename)
        if tool == 'trace_viewer':
          trace = trace_events_pb2.Trace()
          trace.devices[0].name = run
          data = trace.SerializeToString()
        else:
          data = tool.encode('utf-8')
        with open(tool_file, 'wb') as f:
          f.write(data)
  with open(os.path.join(plugin_logdir, 'noise'), 'w') as f:
    f.write('Not a dir, not a run.')


def write_empty_event_file(logdir):
  w = FileWriter(logdir, filename_suffix=EVENT_FILE_SUFFIX)
  w.close()


class ProfilePluginTest(unittest.TestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.logdir = self.tempdir.name
    os.mkdir(f"{self.logdir}/plugins")
    self.plugin = utils.create_profile_plugin(self.logdir)
    self.multiplexer = self.plugin.multiplexer

  def tearDown(self):
    self.tempdir.cleanup()

  def testRuns_logdirWithoutEventFile(self):
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    runs = dict(self.plugin.generate_run_to_tools())
    self.assertSetEqual(frozenset(runs.keys()), frozenset(RUN_TO_TOOLS.keys()))
    self.assertListEqual(runs['foo'], RUN_TO_TOOLS['foo'])
    self.assertListEqual(runs['bar'], [])
    self.assertListEqual(runs['baz'], RUN_TO_TOOLS['baz'])
    self.assertListEqual(runs['qux'], RUN_TO_TOOLS['qux'])
    self.assertListEqual(runs['empty'], [])

  def testRuns_logdirWithEventFIle(self):
    write_empty_event_file(self.logdir)
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    runs = dict(self.plugin.generate_run_to_tools())
    self.assertSetEqual(frozenset(runs.keys()), frozenset(RUN_TO_TOOLS.keys()))

  def testRuns_withSubdirectories(self):
    subdir_a = os.path.join(self.logdir, 'a')
    subdir_b = os.path.join(self.logdir, 'b')
    subdir_b_c = os.path.join(subdir_b, 'c')
    generate_testdata(self.logdir)
    generate_testdata(subdir_a)
    generate_testdata(subdir_b)
    generate_testdata(subdir_b_c)
    write_empty_event_file(self.logdir)
    write_empty_event_file(subdir_a)
    # Skip writing an event file for subdir_b
    write_empty_event_file(subdir_b_c)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    runs = dict(self.plugin.generate_run_to_tools())
    # Expect runs for the logdir root, 'a', and 'b/c' but not for 'b'
    # because it doesn't contain a tfevents file.
    expected = set(RUN_TO_TOOLS.keys())
    expected.update(set('a/' + run for run in RUN_TO_TOOLS.keys()))
    expected.update(set('b/c/' + run for run in RUN_TO_TOOLS.keys()))
    self.assertSetEqual(frozenset(runs.keys()), expected)

  def testHosts(self):
    generate_testdata(self.logdir)
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    hosts = self.plugin.host_impl('foo', 'trace_viewer')
    self.assertListEqual(['host0', 'host1'], hosts)
    hosts_a = self.plugin.host_impl('a/foo', 'trace_viewer')
    self.assertListEqual(['host0', 'host1'], hosts_a)
    hosts_q = self.plugin.host_impl('qux', 'tensorflow_stats')
    self.assertListEqual(hosts_q, [])

  def testData(self):
    generate_testdata(self.logdir)
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    data, _ = self.plugin.data_impl(
        utils.make_data_request('foo', 'trace_viewer', 'host0'))
    trace = json.loads(data)
    self.assertEqual(trace, EXPECTED_TRACE_DATA)
    data, _ = self.plugin.data_impl(
        utils.make_data_request('a/foo', 'trace_viewer', 'host0'))
    trace_a = json.loads(data)
    self.assertEqual(trace_a, EXPECTED_TRACE_DATA)
    data, _ = self.plugin.data_impl(
        utils.make_data_request('qux', 'trace_viewer'))
    trace_qux = json.loads(data)
    expected_trace_qux = copy.deepcopy(EXPECTED_TRACE_DATA)
    expected_trace_qux['traceEvents'][0]['args']['name'] = 'qux'
    self.assertEqual(trace_qux, expected_trace_qux)

    # Invalid tool/run/host.
    data, _ = self.plugin.data_impl(
        utils.make_data_request('foo', 'nonono', 'host0'))
    self.assertIsNone(data)
    data, _ = self.plugin.data_impl(
        utils.make_data_request('bar', 'unsupported', 'host1')
    )
    self.assertIsNone(data)
    # Original tests also asserted these return None, but it seems
    # that our file reading is handled differently than tensorflow's
    # resulting in FileNotFoundError for these examples
    self.assertRaises(
        FileNotFoundError,
        self.plugin.data_impl,
        utils.make_data_request('foo', 'trace_viewer', ''),
    )
    self.assertRaises(
        FileNotFoundError,
        self.plugin.data_impl,
        utils.make_data_request('bar', 'trace_viewer', 'host0'),
    )
    self.assertRaises(
        FileNotFoundError,
        self.plugin.data_impl,
        utils.make_data_request('qux', 'trace_viewer', 'host'),
    )
    self.assertRaises(
        FileNotFoundError,
        self.plugin.data_impl,
        utils.make_data_request('empty', 'trace_viewer', ''),
    )
    self.assertRaises(
        FileNotFoundError,
        self.plugin.data_impl,
        utils.make_data_request('a', 'trace_viewer', ''),
    )

  def testActive(self):

    def wait_for_thread():
      with self.plugin._is_active_lock:
        pass

    # Launch thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Should be false since there's no data yet.
    self.assertFalse(self.plugin.is_active())
    wait_for_thread()
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    # Launch a new thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Now that there's data, this should be active.
    self.assertTrue(self.plugin.is_active())

  def testActive_subdirectoryOnly(self):

    def wait_for_thread():
      with self.plugin._is_active_lock:
        pass

    # Launch thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Should be false since there's no data yet.
    self.assertFalse(self.plugin.is_active())
    wait_for_thread()
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    # Launch a new thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Now that there's data, this should be active.
    self.assertTrue(self.plugin.is_active())

if __name__ == '__main__':
    unittest.main()
